import os
import torch
import numpy as np
import torch.nn.functional as F

def train_decision_transformer_reference(
    model, train_loader, val_loader,
    optimizer, device, num_epochs=50,
    checkpoint_dir="checkpoints"
):
    # Crear carpeta si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    history = {'train_loss': [], 'val_loss': []}

    # Guardar el mejor modelo según validación
    best_val_loss = float("inf")

    warmup_steps = 15
    initial_lr = 1e-5
    final_lr = 1e-4


    for epoch in range(num_epochs):

        # -------- TRAIN MODE ----------
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            states = batch['states'].to(device)      # (B, L)
            actions = batch['actions'].to(device)    # (B, L)
            rtg = batch['rtg'].to(device)            # (B, L, 1)
            timesteps = batch['timesteps'].to(device) # (B, L)
            groups = batch['groups'].to(device)      # (B,)
            targets = batch['targets'].to(device)    # (B, L) - next items


            logits = model(states, actions, rtg, timesteps, groups)

            loss = F.cross_entropy(
                logits.transpose(1, 2),
                targets,
                ignore_index=-1
            )

            optimizer.zero_grad()

             # ==== WARMUP ====
            if epoch < warmup_steps:
                lr = initial_lr + (final_lr - initial_lr) * (epoch / warmup_steps)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        if epoch % 10 == 0:
            # -------- VALIDATION MODE ----------
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    states = batch['states'].to(device)
                    actions = batch['actions'].to(device)
                    rtg = batch['rtg'].to(device)
                    timesteps = batch['timesteps'].to(device)
                    groups = batch['groups'].to(device)
                    targets = batch['targets'].to(device)

                    logits = model(states, actions, rtg, timesteps, groups)

                    loss = F.cross_entropy(
                        logits.transpose(1, 2),
                        targets,
                        ignore_index=-1
                    )

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # -------- CHECKPOINTING ----------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"dt_epoch{epoch+1}_val{avg_val_loss:.4f}.pt"
                )

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "train_loss": avg_train_loss,
                }, checkpoint_path)

                print(f"  ✔ Checkpoint guardado: {checkpoint_path}")

            # -------- LOG ----------
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.4f}")

    return model, history
