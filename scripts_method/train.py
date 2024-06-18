import os.path as op
import sys
from pprint import pformat

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint#, ModelSummary
from pytorch_lightning.utilities.model_summary import summarize

sys.path.append(".")

import common.comet_utils as comet_utils
import src.factory as factory
from common.torch_utils import reset_all_seeds
from src.utils.const import args


def main(args):
    if args.get("experiment") is not None:
        comet_utils.log_exp_meta(args)
    reset_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = factory.fetch_model(args)#.to(device)
    if args.load_ckpt != "":
        ckpt = torch.load(args.load_ckpt)
        wrapper.load_state_dict(ckpt["state_dict"])
        logger.info(f"Loaded weights from {args.load_ckpt}")

    # wrapper.model.arti_head.object_tensors.to(device)

    ckpt_callback = ModelCheckpoint(
        monitor="loss__val",
        verbose=True,
        save_top_k=5,
        mode="min",
        every_n_epochs=args.eval_every_epoch,
        save_last=True,
        dirpath=op.join(args.log_dir, "checkpoints"),
    )

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)

    # model_summary_cb = ModelSummary(max_depth=3)
    print(summarize(lightning_module=wrapper, max_depth=3))
    callbacks = [ckpt_callback, pbar_cb] # ,model_summary_cb]
    trainer = pl.Trainer(
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.acc_grad,
        devices=args.num_gpus,
        # strategy="ddp_find_unused_parameters_true",
        strategy="fsdp" if args.num_gpus > 1 else "auto",
        accelerator="gpu",
        logger=None,
        min_epochs=args.num_epoch,
        max_epochs=args.num_epoch,
        callbacks=callbacks,
        log_every_n_steps=args.log_every,
        default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.eval_every_epoch,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )

    reset_all_seeds(args.seed)
    train_loader = factory.fetch_dataloader(args, "train")
    logger.info(f"Hyperparameters: \n {pformat(args)}")
    logger.info("*** Started training ***")
    reset_all_seeds(args.seed)
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    val_loaders = [factory.fetch_dataloader(args, "val")]
    wrapper.set_training_flags()  # load weights if needed
    trainer.fit(wrapper, train_loader, val_loaders, ckpt_path=ckpt_path)
    if args.num_gpus > 1:
        trainer.save_checkpoint("distributed_model.ckpt")


def just_clear_log():
    if not args.clear_log:
        return
    import os
    os.rmdir(os.path.join("logs",args.exp_key))
    logger.info("Logs cleared")

if __name__ == "__main__":
    try:
        main(args)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory. Please reduce batch size or image size.")
            import os
            just_clear_log()
            os._exit(1)
        else:
            raise e
    finally:
        just_clear_log()

