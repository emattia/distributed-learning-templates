from metaflow import FlowSpec, step, parallel
from metaflow.profilers import gpu_profile
N_WORKERS = 2


class Llama8bTorchtuneMultinodeFinetuning(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_WORKERS)

    @gpu_profile(interval=1)
    @parallel
    @step
    def train(self):
        import subprocess
        # torchtune proc on train.py
        # from best checkpoint
            # torch.compile
            # save full fp16 or bf16
            # [optional] produce .gguf
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    Llama8bTorchtuneMultinodeFinetuning()