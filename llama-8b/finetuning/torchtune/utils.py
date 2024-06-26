from typing import Literal, Union

GATED_HF_ORGS = [
    "meta-llama",
]


def get_all_recipes():
    for recipe in get_all_recipes():
        # If there are no configs for a recipe, print a blank config
        recipe_str = recipe.name
        if len(recipe.configs) == 0:
            row = f"{recipe_str:<40} {self.NULL_VALUE:<40}"
            print(row)
        for i, config in enumerate(recipe.configs):
            # If there are multiple configs for a single recipe, omit the recipe name
            # on latter configs
            if i > 0:
                recipe_str = ""
            row = f"{recipe_str:<40} {config.name:<40}"
            print(row)


# class Params(object):

# # https://github.com/pytorch/torchtune/blob/15c918d65d79e03abcbd0c5e94d2b116bd368412/torchtune/_cli/download.py#L57
# hf_repo_id = Parameter('-r', '--repo-id', default='meta-llama/Meta-Llama-3-8B-Instruct')
# output_dir = Parameter('-o', '--output-dir', default="output_dir")
# output_dir_use_symlinks = Parameter('-s', '--symlink-output-dir', type=Union[Literal["auto"], bool], default=False)
# ignore_patterns = Parameter('-i', '--ignore-patterns', default='*.safetensors', help=
#     "If provided, files matching any of the patterns are not downloaded. Defaults to ignoring "
#     "safetensors files to avoid downloading duplicate weights.")

# # data
# workflow = Parameter('-w', '--workflow', help='What type of workflow / recipe?', default='lora_finetune_single_device') # recipe
# config = Parameter('-c', '--config', help='Torchtune config', default='llama3/8B_lora_single_device')


class GatedRepoError(Exception):
    """Exception raised for errors related to gated repositories."""

    def __init__(self, repo_id):
        self.message = (
            f"Access to {repo_id} needs the HF_TOKEN environment variable set."
        )
        super().__init__(self.message)
