# Running on TPUs

Currently we show how to run `tx` on a single TPU machine.

## Setting up the TPU VM

We use [queued resources](https://cloud.google.com/tpu/docs/queued-resources) to start the VM since it seems to be the recommended way:

```bash
gcloud compute tpus queued-resources create <TPU_PREFIX> --project=<PROJECT> --zone=<ZONE> --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --node-count=1 --node-prefix=<TPU_PREFIX> --scopes=https://www.googleapis.com/auth/cloud-platform.read-only,https://www.googleapis.com/auth/devstorage.read_write --network=<NETWORK> --subnetwork=<SUBNETWORK> --spot
```

After the VM is started, you can ssh into it via

```bash
gcloud compute tpus tpu-vm ssh <TPU_PREFIX>-0
```

## Starting the training

Once you are logged into the VM, install `uv` and clone the repository with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/tx-project/tx
cd tx
```

You can then start the training with

```bash

```

See more the full set of options in [the CLI reference](../reference.md).