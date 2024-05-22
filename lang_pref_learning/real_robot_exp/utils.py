import re
import cv2
import torch
import numpy as np
from einops import rearrange

from data.utils import WidowX_OBJECT_STATE_DIM


def get_traj_embeds_wx(trajs, model, device, use_img_obs=False, img_obs=None):
    """
    Get the trajectory embeddings for the given trajs

    Args:
        - trajs (np.ndarray): Trajectories to get embeddings for
        - model (nn.Module): Model to extract embeddings
        - device (torch.device): Device to use for computation
        - use_img_obs (bool): Whether to use image observations
        - img_obs (np.ndarray): Image observations for the trajs
    
    Returns:
        - traj_embeds (np.ndarray): Trajectory embeddings
    """

    # Process the trajs in batch 1
    trajs_embeds = []
    for i in range(0, len(trajs)):
        batch_inputs = {}
        traj, traj_img_obs = trajs[i], img_obs[i]
        if use_img_obs:
            traj_img_obs = rearrange(traj_img_obs, 't h w c -> t c h w')
            
            batch_inputs['img_obs'] = torch.from_numpy(traj_img_obs).unsqueeze(0).float().to(device)
            batch_inputs['states'] = torch.from_numpy(
                traj[:, WidowX_OBJECT_STATE_DIM: ]
                ).unsqueeze(0).float().to(device)
            
        batch_encoded_trajs = model.traj_encoder(batch_inputs)
        batch_trajs_embeds = torch.mean(batch_encoded_trajs, dim=-2, keepdim=False).detach().cpu().numpy()
        trajs_embeds.append(batch_trajs_embeds)

    traj_embeds = np.concatenate(trajs_embeds, axis=0)

    return traj_embeds



def replay_trajectory_video(traj_images, title='Current Trajectory', frame_rate=10, close=True):
    """
    Replay a trajectory from a list of images

    :param traj_images: the list of images
    """
    n_frames, h, w, c = traj_images.shape

    # resize the images to a common size

    for i in range(n_frames):
        frame = traj_images[i]

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow(title, frame)
        cv2.moveWindow(title, 800, 600)

        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):  # Press 'q' to quit the playback
            break
    
    if close:
        cv2.destroyAllWindows() 



def remove_special_characters(input_string):
    # Define the pattern to match special characters (non-alphanumeric and non-whitespace)
    pattern = r'[^a-zA-Z0-9\s]'
    # Use re.sub() to replace the special characters with an empty string
    cleaned_string = re.sub(pattern, '', input_string)
    cleaned_string = cleaned_string.strip()
    cleaned_string += "."
    return cleaned_string


def get_lang_embed(nlcomp, model, device, tokenizer, preprocessed=False, lang_model=None):
    if preprocessed:
        assert lang_model is not None
        inputs = tokenizer(nlcomp, return_tensors="pt")
        lang_outputs = lang_model(**inputs)
        embedding = lang_outputs.last_hidden_state

        # Average across the sequence to get a sentence-level embedding
        embedding = torch.mean(embedding, dim=1, keepdim=False)
        lang_embed = model.lang_encoder(embedding.to(device)).squeeze(0).detach().cpu().numpy()

    else:
        # First tokenize the NL comparison and get the embedding
        inputs = tokenizer(nlcomp, return_tensors="pt")
        # move inputs to the device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden_states = model.lang_encoder(**inputs).last_hidden_state
        lang_embed = torch.mean(hidden_states, dim=1, keepdim=False).squeeze(0).detach().cpu().numpy()

    return lang_embed