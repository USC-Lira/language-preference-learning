import re
import cv2
import time
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


def replay_traj_widowx(env, policy_dict):
    actions = np.stack([d['actions'] for d in policy_dict], axis=0)

    last_tstep = time.time()

    env._controller.open_gripper(True)
    env.move_to_neutral()

    for action in actions:
        env.step(action)
        # t = time.time()
        # while True:
        #     if t - last_tstep > env_params['move_duration']:
        #         print(f'loop {t - last_tstep}s')
        #         obs = env.step(action)
        #         obs_imgs.append(obs['images'])
        #         last_tstep = t
        #         break


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


def get_traj_lang_embeds(trajs, nlcomps, model, device, use_bert_encoder, 
                         tokenizer=None, nlcomps_bert_embeds=None,
                         use_img_obs=False, img_obs=None):
    """
    Get the trajectory and language embeddings for the given trajs and nlcomps

    Args:
        - trajs (np.ndarray): Trajectories to get embeddings for
        - nlcomps (List[str]): Natural language comparisons to get embeddings for
        - model (nn.Module): Model to extract embeddings
        - device (torch.device): Device to use for computation
        - use_bert_encoder (bool): Whether to use BERT for language encoding
        - tokenizer (BertTokenizer): Tokenizer to use for BERT encoding
        - nlcomps_bert_embeds (List[np.ndarray]): BERT embeddings for the NL comparisons
        - use_img_obs (bool): Whether to use image observations
        - img_obs (np.ndarray): Image observations for the trajs
        - actions (np.ndarray): Actions for the trajs
    
    Returns:
        - traj_embeds (np.ndarray): Trajectory embeddings
        - lang_embeds (np.ndarray): Language embeddings
    """
    traj_embeds = get_traj_embeds_wx(trajs, model, device, use_img_obs, img_obs)

    # Get the nearest trajectory embedding for each language comparison
    lang_embeds = []
    if use_bert_encoder:
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

    else:
        assert nlcomps_bert_embeds is not None
        for i, nlcomp_bert_embed in enumerate(nlcomps_bert_embeds):
            lang_embed = model.lang_encoder(
                torch.from_numpy(nlcomp_bert_embed).float().to(device)).detach().cpu().numpy()
            lang_embeds.append(lang_embed)

    lang_embeds = np.array(lang_embeds)

    return traj_embeds, lang_embeds


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