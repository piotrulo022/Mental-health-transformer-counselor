from model import build_transformer
from ds_tokenizer import *
from tqdm import tqdm
import os

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)    
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for target
        dec_mask = decoder_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, dec_mask)
        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            # print('OUTPUT Ids', model_out)
            # print('Truth ids:', batch['decoder_input'].detach().cpu().numpy())
            # print('OUTPUT TEXT', model_out_text)
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print('-'*80)
            print(f"{f'SOURCE: ':>12}{source_text}")
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print('-'*console_width)
                break


def train_model(model, num_epochs, device, train_loader, val_loader, loss_fn, optimizer, tokenizer_tgt, tgt_seq_len, output_path):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print('Epoch: ', epoch + 1)
        model.train()

        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_loader, desc = f'Processing epoch: {epoch:02d}')
        for batch in batch_iterator:
        # for batch in train_loader:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            # print(decoder_output)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            
            loss.backward() # jak to przez to nie działa no to jaja
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            

        # validation loop
            run_validation(model, val_loader, tokenizer_tgt,  tgt_seq_len, device)



    print(f'Model trained successfully! Saving to {output_path}')
    try:
        torch.save(model.state_dict(), output_path)
    except Exception as e:
        print(e)
        torch.save(model.state_dict(), 'model_state_dict.pt')


