# 集合通信常用框架

主流框架：MPI gloo NCCL

## 简单比较

reference from: [Communication Backends, Raw performance benchmarking](https://mlbench.github.io/2020/09/08/communication-backend-comparison/)

| Backend   | Comm. Functions                               | Optimized For | Float32   | Float64   |
| ---       | ---                                           | ---           | ---       | ---       |
| MPI       | All                                           | CPU, GPU      | Y         | N         |
| GLOO      | All (on CPU), bcast & allreduce (on GPU)      | CPU           | Y         | Y         |
| NCCL      | bcast, allreduce, reduce and alltogather      | GPU only      | Y         | Y         |

## pytorch文档

reference from: [DISTRIBUTED COMMUNICATION PACKAGE - TORCH.DISTRIBUTED](https://pytorch.org/docs/stable/distributed.html#distributed-communication-package-torch-distributed)

### 简单总结文档

- CPU上使用Gloo
- GPU上使用NCCL