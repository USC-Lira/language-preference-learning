import numpy as np
import torch

from einops import rearrange

from data.utils import RS_OBJECT_STATE_DIM, RS_PROPRIO_STATE_DIM, WidowX_PROPRIO_STATE_DIM, WidowX_OBJECT_STATE_DIM


def get_traj_lang_embeds(trajs, nlcomps, model, device, 
                         tokenizer=None, 
                         use_img_obs=False, img_obs=None, actions=None,
                         traj_encoder_type="mlp"):
    """
    Get the trajectory and language embeddings for the given trajs and nlcomps

    Args:
        - trajs (np.ndarray): Trajectories to get embeddings for
        - nlcomps (List[str]): Natural language comparisons to get embeddings for
        - model (nn.Module): Model to extract embeddings
        - device (torch.device): Device to use for computation
        - tokenizer (BertTokenizer): Tokenizer to use for BERT encoding
        - use_img_obs (bool): Whether to use image observations
        - img_obs (np.ndarray): Image observations for the trajs
        - actions (np.ndarray): Actions for the trajs
    
    Returns:
        - traj_embeds (np.ndarray): Trajectory embeddings
        - lang_embeds (np.ndarray): Language embeddings
    """
    trajs_inputs = {
        'trajs': torch.from_numpy(trajs).float().to(device),
    }
    if use_img_obs:
        if img_obs.shape[-1] == 3:
            img_obs = rearrange(img_obs, 'b t h w c -> b t c h w')

        trajs_inputs['img_obs'] = torch.from_numpy(img_obs).float()
        trajs_inputs['states'] = torch.from_numpy(
            trajs[:, :, RS_OBJECT_STATE_DIM:]
            ).float()

    # Process the trajs in batches
    trajs_embeds = []
    batch_size = 8
    for i in range(0, len(trajs), batch_size):
        batch_trajs_inputs = {k: v[i:i + batch_size].to(device) for k, v in trajs_inputs.items()}
        if traj_encoder_type == "cnn":
            batch_encoded_trajs = model.traj_encoder(batch_trajs_inputs)
        else:
            batch_encoded_trajs = model.traj_encoder(batch_trajs_inputs['trajs'])
        batch_trajs_embeds = torch.mean(batch_encoded_trajs, dim=-2, keepdim=False).detach().cpu().numpy()
        trajs_embeds.append(batch_trajs_embeds)

    traj_embeds = np.concatenate(trajs_embeds, axis=0)

    # Get the nearest trajectory embedding for each language comparison
    lang_embeds = []
    assert tokenizer is not None
    lang_inputs = tokenizer(
        nlcomps,
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    lang_inputs = {k: v.to(device) for k, v in lang_inputs.items()}
    lang_outputs = model.lang_encoder(**lang_inputs)
    emebddings = lang_outputs.last_hidden_state
    lang_embeds = torch.mean(emebddings, dim=1, keepdim=False).detach().cpu().numpy()

    lang_embeds = np.array(lang_embeds)

    return traj_embeds, lang_embeds


def get_lang_embed(nlcomp, model, device, tokenizer, lang_model_name=None):
    # First tokenize the NL comparison and get the embedding
    inputs = tokenizer(nlcomp, return_tensors="pt")
    # move inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden_states = model.lang_encoder(**inputs).last_hidden_state
    lang_embed = torch.mean(hidden_states, dim=1, keepdim=False).squeeze(0).detach().cpu().numpy()

    return lang_embed
