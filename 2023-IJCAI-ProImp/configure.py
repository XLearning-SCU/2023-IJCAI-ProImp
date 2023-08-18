def get_default_config(data_name):
    if data_name in ['Scene_15']:#feature
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 256],
                arch2=[59, 1024, 1024, 1024, 256],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=0,
                batch_size=1024,
                epoch=150,
                lr=1e-3,
                num=15,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name in ['NoisyMNIST']:#row
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[784, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.0,
                seed=0,
                batch_size=1024,
                epoch=150,
                lr=1e-3,
                num=10,
                dim=128,
                pretrain_epoch=50
            ),
        )

    elif data_name in ['Reuters']:#feature
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[10, 32, 1024, 1024, 32],
                arch2=[10, 32, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=0,
                batch_size=1024,
                epoch=150,
                lr=1e-3,
                num=6,
                dim=32,
                pretrain_epoch=50
            ),
        )

    elif data_name in ['MNIST-USPS']:#raw
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[256, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=0,
                batch_size=1024,
                epoch=150,
                lr=1e-3,
                num=10,
                dim=128,
                pretrain_epoch=50
            ),
        )

    elif data_name in ['cub_googlenet']:#feature
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[1024, 512, 1024, 1024, 256],  # the last number is the dimension of latent representation
                arch2=[300, 512, 1024, 1024, 256],
                activations1='sigmoid',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=0,
                batch_size=1024,
                epoch=150,
                lr=1e-3,
                num=10,
                dim=256,
                pretrain_epoch=50
            ),
        )

    else:
        raise Exception('Undefined data_name')
