from physilearning.train import Trainer

for i in range(1, 11):
    trainer = Trainer('config.yaml')
    trainer.model_save_prefix = f'20240913_fitness_{i}'
    trainer.learn()






