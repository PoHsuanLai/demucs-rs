//! Guest-side `RunnerChannel` impl that ships burn ops to the host
//! through the WIT `burn` interface.
//!
//! Mirrors `burn-remote::client::RemoteChannel` 1:1 — same atomic
//! `TensorId` counter, same single-backend bridge (panic on
//! `change_backend_*` since we only have one device), same async
//! `read_tensor_async` returning a `DynFut`.
//!
//! The host implements all the heavy lifting; this file is *only* the
//! transport adapter.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_router::{MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient};
use burn_std::backtrace::BackTrace;
use burn_std::future::DynFut;
use burn_tensor::backend::{DTypeUsageSet, Device, DeviceId, DeviceOps, ExecutionError};
use burn_tensor::{BoolStore, DType, Shape, TensorData};

use crate::dawai::extension::burn as host;

// =============================================================================
// Device
// =============================================================================

/// Single-device handle — the host owns exactly one runner, so this is
/// effectively a unit. The `id` is the host's reported device id at
/// init time; kept around for `DeviceOps::id()`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WitDevice {
    id: u16,
}

impl Default for WitDevice {
    fn default() -> Self {
        let info = host::device();
        // Host wire is `u32` but DeviceId is `u16`; truncate. The
        // host only ever runs one runner, so any id beyond `u16::MAX`
        // collisions would already mean broken host setup.
        Self {
            id: info.id as u16,
        }
    }
}

impl Device for WitDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            id: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.id,
        }
    }
}

impl DeviceOps for WitDevice {}

// =============================================================================
// Client
// =============================================================================

/// Atomic id counter — matches `burn-remote::RemoteSender::new_tensor_id`.
/// Burn ops embed the ids they expect inside `OperationIr`, so the
/// router just needs every id to be unique within this client.
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

fn new_tensor_id() -> TensorId {
    TensorId::new(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Clone)]
pub struct WitClient {
    device: WitDevice,
    // Pulls in `Arc` so the type isn't trivially `Copy`; the router
    // clones clients liberally.
    _refs: Arc<()>,
}

impl WitClient {
    fn new(device: WitDevice) -> Self {
        Self {
            device,
            _refs: Arc::new(()),
        }
    }
}

impl RunnerClient for WitClient {
    type Device = WitDevice;

    fn register_op(&self, op: OperationIr) {
        let bytes = rmp_serde::to_vec(&op).expect("serialize OperationIr");
        host::register_op(&bytes).expect("register_op host call");
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let id = tensor.id.value();
        let shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();
        let dtype = encode_dtype(tensor.dtype);
        let result = match host::read_tensor(id, &shape, dtype) {
            Ok(td) => {
                let dtype_back = decode_dtype(td.dtype).unwrap_or(DType::F32);
                let shape_usize: Vec<usize> = td.shape.iter().map(|&d| d as usize).collect();
                Ok(TensorData::from_bytes_vec(td.bytes, shape_usize, dtype_back))
            }
            Err(e) => Err(ExecutionError::Generic {
                reason: format!("read_tensor (host): {e}"),
                backtrace: BackTrace::capture(),
            }),
        };
        Box::pin(async move { result })
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let id = new_tensor_id();
        let shape = Shape::from(data.shape.clone());
        let dtype = data.dtype;
        let wire = host::TensorData {
            dtype: encode_dtype(dtype),
            shape: shape.iter().map(|&d| d as u32).collect(),
            bytes: data.into_bytes().to_vec(),
        };
        host::register_tensor_data(id.value(), &wire).expect("register_tensor_data host call");
        RouterTensor::new(id, shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        host::sync().map_err(|e| ExecutionError::Generic {
            reason: format!("sync (host): {e}"),
            backtrace: BackTrace::capture(),
        })
    }

    fn seed(&self, seed: u64) {
        host::seed(seed);
    }

    fn create_empty_handle(&self) -> TensorId {
        new_tensor_id()
    }

    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        let bits = host::dtype_usage(encode_dtype(dtype));
        DTypeUsageSet::from_u32_truncated(bits)
    }
}

// =============================================================================
// Channel + Bridge
// =============================================================================

#[derive(Clone, Debug)]
pub struct WitChannel {
    _private: (),
}

pub struct WitBridge;

impl MultiBackendBridge for WitBridge {
    type TensorHandle = WitTensorHandle;
    type Device = WitDevice;

    fn change_backend_float(
        _tensor: Self::TensorHandle,
        _shape: Shape,
        _target_device: &Self::Device,
    ) -> Self::TensorHandle {
        unreachable!("WitBridge::change_backend_float — single-backend router")
    }

    fn change_backend_int(
        _tensor: Self::TensorHandle,
        _shape: Shape,
        _target_device: &Self::Device,
    ) -> Self::TensorHandle {
        unreachable!("WitBridge::change_backend_int — single-backend router")
    }

    fn change_backend_bool(
        _tensor: Self::TensorHandle,
        _shape: Shape,
        _target_device: &Self::Device,
    ) -> Self::TensorHandle {
        unreachable!("WitBridge::change_backend_bool — single-backend router")
    }
}

pub struct WitTensorHandle {
    _tensor: TensorIr,
    _client: WitClient,
}

impl RunnerChannel for WitChannel {
    type Device = WitDevice;
    type Bridge = WitBridge;
    type Client = WitClient;
    type FloatElem = f32;
    type IntElem = i32;
    type BoolElem = u32;

    fn name(_device: &Self::Device) -> String {
        "wit".to_string()
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        WitClient::new(device.clone())
    }

    fn get_tensor_handle(
        tensor: &TensorIr,
        client: &Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorHandle {
        WitTensorHandle {
            _tensor: tensor.clone(),
            _client: client.clone(),
        }
    }

    fn register_tensor(
        _client: &Self::Client,
        _handle: <Self::Bridge as MultiBackendBridge>::TensorHandle,
        _shape: Shape,
        _dtype: DType,
    ) -> RouterTensor<Self::Client> {
        unreachable!("WitChannel::register_tensor — single-backend router")
    }
}

// =============================================================================
// Convenience type aliases
// =============================================================================

/// The `Backend` alias extension code uses.
pub type Backend = burn_router::BackendRouter<WitChannel>;

// =============================================================================
// dtype encode / decode (mirrors host)
// =============================================================================

fn encode_dtype(d: DType) -> u8 {
    match d {
        DType::F64 => 0,
        DType::F32 => 1,
        DType::Flex32 => 2,
        DType::F16 => 3,
        DType::BF16 => 4,
        DType::I64 => 5,
        DType::I32 => 6,
        DType::I16 => 7,
        DType::I8 => 8,
        DType::U64 => 9,
        DType::U32 => 10,
        DType::U16 => 11,
        DType::U8 => 12,
        DType::Bool(_) => 13,
        DType::QFloat(_) => 14,
    }
}

fn decode_dtype(b: u8) -> Option<DType> {
    Some(match b {
        0 => DType::F64,
        1 => DType::F32,
        2 => DType::Flex32,
        3 => DType::F16,
        4 => DType::BF16,
        5 => DType::I64,
        6 => DType::I32,
        7 => DType::I16,
        8 => DType::I8,
        9 => DType::U64,
        10 => DType::U32,
        11 => DType::U16,
        12 => DType::U8,
        13 => DType::Bool(BoolStore::Native),
        _ => return None,
    })
}
