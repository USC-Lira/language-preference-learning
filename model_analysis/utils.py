import numpy as np
import torch


def get_traj_lang_embeds(trajs, nlcomps, model, device, use_bert_encoder, tokenizer=None, nlcomps_bert_embeds=None):
    encoded_trajs = model.traj_encoder(torch.from_numpy(trajs).float().to(device)).detach().cpu().numpy()
    traj_embeds = np.mean(encoded_trajs, axis=-2, keepdims=False)

    # Get the nearest trajectory embedding for each language comparison

    lang_embeds = []
    if use_bert_encoder:
        assert tokenizer is not None
        for i, nlcomp in enumerate(nlcomps):
            # First tokenize the NL comparison and get the embedding
            tokens = tokenizer.tokenize(tokenizer.cls_token + " " + nlcomp + " " + tokenizer.sep_token)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # Pad sequences to the common length
            padding_length = 64 - len(token_ids)
            # Create attention mask
            attention_mask = [1] * len(token_ids) + [0] * padding_length
            token_ids += [tokenizer.pad_token_id] * padding_length
            token_ids = torch.from_numpy(np.asarray(token_ids)).unsqueeze(0).to(device)
            attention_mask = torch.from_numpy(np.asarray(attention_mask)).unsqueeze(0).to(device)
            hidden_states = model.lang_encoder(token_ids, attention_mask=attention_mask).last_hidden_state
            lang_embed = torch.mean(hidden_states, dim=1, keepdim=False).detach().cpu().numpy()
            lang_embeds.append(lang_embed)

    else:
        assert nlcomps_bert_embeds is not None
        for i, nlcomp_bert_embed in enumerate(nlcomps_bert_embeds):
            lang_embed = model.lang_encoder(
                torch.from_numpy(nlcomp_bert_embed).float().to(device)).detach().cpu().numpy()
            lang_embeds.append(lang_embed)

    lang_embeds = np.array(lang_embeds)

    return traj_embeds, lang_embeds


def get_lang_embed(nlcomp, model, device, tokenizer, use_bert_encoder=False, bert_model=None):
    if not use_bert_encoder:
        assert bert_model is not None
        inputs = tokenizer(nlcomp, return_tensors="pt")
        bert_output = bert_model(**inputs)
        embedding = bert_output.last_hidden_state

        # Average across the sequence to get a sentence-level embedding
        embedding = torch.mean(embedding, dim=1, keepdim=False)
        lang_embed = model.lang_encoder(embedding.to(device)).squeeze(0).detach().cpu().numpy()

    else:
        # First tokenize the NL comparison and get the embedding
        tokens = tokenizer.tokenize(tokenizer.cls_token + " " + nlcomp + " " + tokenizer.sep_token)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Pad sequences to the common length
        padding_length = 64 - len(token_ids)
        # Create attention mask
        attention_mask = [1] * len(token_ids) + [0] * padding_length
        token_ids += [tokenizer.pad_token_id] * padding_length
        token_ids = torch.from_numpy(np.asarray(token_ids)).unsqueeze(0).to(device)
        attention_mask = torch.from_numpy(np.asarray(attention_mask)).unsqueeze(0).to(device)
        hidden_states = model.lang_encoder(token_ids, attention_mask=attention_mask).last_hidden_state
        lang_embed = torch.mean(hidden_states, dim=1, keepdim=False).squeeze(0).detach().cpu().numpy()

    return lang_embed
