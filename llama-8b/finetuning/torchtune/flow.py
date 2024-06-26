from metaflow import FlowSpec, step, secrets, kubernetes, resources, parallel, Parameter
from metaflow.profilers import gpu_profile
from utils import *
import subprocess
import os

class TorchtuneMultinodeBase(FlowSpec):

    # https://github.com/pytorch/torchtune/blob/15c918d65d79e03abcbd0c5e94d2b116bd368412/torchtune/_cli/download.py#L57
    hf_repo_id = Parameter('repo-id', default='meta-llama/Meta-Llama-3-8B-Instruct')

    # torchtune comes with these:  
    dataset = Parameter('data', help='Which dataset to use', default='alpaca_dataset')
    # base types include: PackedDataset, ConcatDataset, TextCompletionDataset, ChatDataset, InstructionDataset, PreferenceDataset
    # example: alpaca_dataset is an InstructionDataset, slimorca_dataset is a ChatDataset.

    local_checkpoint_in_path = Parameter('in-chkpt', help='Where to store the checkpoint locally?', default='/tmp/checkpoint-in')
    local_checkpoint_out_path = Parameter('out-chkpt', help='Where to store the checkpoint locally?', default='/tmp/checkpoint-out')

    # https://pytorch.org/torchtune/stable/deep_dives/recipe_deepdive.html#recipe-deepdive
    workflow = Parameter('workflow', help='What type of workflow / recipe?', default='lora_finetune_distributed') 
    # https://pytorch.org/torchtune/stable/deep_dives/configs.html#config-tutorial-label
    config = Parameter('config', help='Torchtune config', default='llama3/8B_lora')
    # To see all combinations of recipe and config, run `tune ls`. 

    @secrets(sources=['huggingface-token'])
    @kubernetes(image='docker.io/eddieob/torchtune:latest', memory=16000)
    @step
    def start(self):
        """
        Check that the necessary credentials are accessible in @secrets and config values are sensible.
        """
        # Check the dataset is legit.
        from torchtune import datasets as tt_datasets
        assert self.dataset in tt_datasets.__all__, f"Choose a dataset from this list: {tt_datasets.__all__}"

        # Huggingface repos routes are like 'org/model'.
        if self.hf_repo_id.split('/')[0] in GATED_HF_ORGS and 'HF_TOKEN' not in os.environ:
            raise GatedRepoError(self.hf_repo_id)
        self.next(self.tune, num_parallel=2)

    @secrets(sources=['huggingface-token'])
    @gpu_profile(interval=1)
    @parallel
    @kubernetes(gpu=4, cpu=12, memory=60000, image='docker.io/eddieob/torchtune:latest')
    @step
    def tune(self):
        """
        Download the data and run a workflow
        """
        # Get the model.
        # https://github.com/pytorch/torchtune/blob/15c918d65d79e03abcbd0c5e94d2b116bd368412/torchtune/_cli/download.py#L126C13-L132C14
        self.download_cmd = [
            "tune", "download", self.hf_repo_id,
                "--hf-token", os.environ.get('HF_TOKEN'),
                "--output-dir", self.local_checkpoint_in_path
        ]
        is_success, stderr = self.exec(self.download_cmd)
        if not is_success:
            raise Exception(stderr)
        
        # Do the tuning job.
        self.run_cmd = [
            "tune", "run",
            "--nproc_per_node", "4", # p3.8xlarge in ob compute pool has 4 GPU cards.
            "--master_port=25678",
            self.workflow, 
            "--config", self.config, 
                f"dataset=torchtune.datasets.{self.dataset}",
                f"tokenizer.path={self.local_checkpoint_in_path}/original/tokenizer.model",
                f"checkpointer.checkpoint_dir={self.local_checkpoint_in_path}/original",
                f"checkpointer.output_dir={self.local_checkpoint_out_path}/new/",
                "batch_size=2"
        ]
        success, stderr = self.exec(self.run_cmd)
        if not success:
            raise Exception(stderr)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

    def exec(self, cmd):
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        ) as proc:
            while proc.poll() is None:
                stdout = proc.stdout.read1()
                try:
                    text = stdout.decode('utf-8')
                except UnicodeDecodeError:
                    text = ''
                print(text, end='', flush=True)
            if proc.returncode != 0:
                if proc.stderr is not None:
                    return False, proc.stderr.read().decode("utf-8")
                else:
                    return False, None
            return True, None

if __name__ == '__main__':
    TorchtuneMultinodeBase()