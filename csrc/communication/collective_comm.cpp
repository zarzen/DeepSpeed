#include <torch/extension.h>

#include <c10d/ProcessGroup.hpp> 
#include <c10d/ProcessGroupNCCL.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <nccl.h>
#include <unordered_map>
#include <vector>

// recording created ncclComm_t
// using processGroup Name as key
std::unordered_map<std::string, ncclComm_t> group_communicators;

void check_tensors(std::vector<at::Tensor>& outputTensors,
                    std::vector<at::Tensor>& inputTensors,
                    int world_size) {
  if (inputTensors.size() == 0 || outputTensors.size() == 0) {
    TORCH_CHECK(false, "output/input tensor list must be nonempty");
  }
  if (outputTensors.size() != inputTensors.size()) {
    TORCH_CHECK(false, "output and input tensors must have same size");
  }

  for (size_t i = 0; i < inputTensors.size(); ++i) {
    auto out = outputTensors[i];
    auto in = inputTensors[i];
    if (out.numel() != in.numel() * world_size) {
      std::stringstream ss;
      ss << "output tensor numel != input tensor numel * world_size at" << i ;
      TORCH_CHECK(false, ss.str());
    }
  }

}

// rank0 create the ncclUniqueId 
// broadcast using old ProcessGroupNCCL
// ncclCommInitRank with ncclUniqueId and same rank and world size from current
// ProcessGroupNCCL
// 
// Note: reason for creating new ncclComm_t, ::c10d::ProcessGroupNCCL didn't expose
// APIs for getting communicator
ncclComm_t create_communicator(std::vector<at::Tensor>& inputTensors,
                                std::string& pg_name, 
                                ::c10d::ProcessGroupNCCL& pg) {
  int rank = pg.getRank();
  int world_size = pg.getSize();
  at::Tensor& first_tensor = inputTensors[0];
  auto device_idx = first_tensor.get_device();

  // 
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;

  auto id_tensor_option =
    torch::TensorOptions()
      .dtype(torch::kUInt8)
      .layout(torch::kStrided) // dense tensor
      .device(torch::kCUDA, device_idx)
      .requires_grad(false);

  std::vector<at::Tensor> bcast_tensor;
  if (rank == 0) {
    auto _result = ncclGetUniqueId(&nccl_id);
    if (_result != ncclSuccess) {
      TORCH_CHECK(false, "Getting nccl unique id failed");
      // it suppose to exit
    } 

    at::Tensor id_tensor = torch::from_blob(&nccl_id, {sizeof(ncclUniqueId)}, id_tensor_option);
    bcast_tensor.push_back(std::move(id_tensor));
  } else {
    at::Tensor id_tensor = torch::empty(sizeof(ncclUniqueId), id_tensor_option).zero_();
    bcast_tensor.push_back(std::move(id_tensor));
  }

  // TODO: 
  // bcast
  auto work = pg.broadcast(bcast_tensor);
  work->wait();

  // if rank != 0 
  // then need to copy ncclUniqueId from bcast_tensor
  // TODO: 

}

// get communicator from global map
// if not found, create a new one
ncclComm_t get_communicator(std::vector<at::Tensor>& inputTensors, 
                            std::string& pg_name, ::c10d::ProcessGroupNCCL& pg) {
  auto found = group_communicators.find(pg_name);
  if (found == group_communicators.end()) {
    return create_communicator(pg_name, pg);
  } else {
    return found->second;
  }
}

int inplaceAllgather(std::vector<at::Tensor>& outputTensors,
                     std::vector<at::Tensor>& inputTensors,
                     ::c10d::ProcessGroupNCCL& pg,
                     std::string pg_name
                     ) {
  // ::c10d::ProcessGroup& p_pg = pg;
  printf("inplaceAllgather:: process group rank %d, size %d, pg_name %s \n", 
        pg.getRank(), pg.getSize(), pg_name.c_str());

  check_tensors(outputTensors, inputTensors, pg.getSize());

  return -1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("inplace_allgather", &inplaceAllgather, "inplace all-gather (without memcpy)");
}
