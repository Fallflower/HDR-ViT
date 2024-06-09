from train import train
from assistant import Options, HelpSave


def main(model):
    options = Options()
    opt = options.load_options(f'./models/{model}/options.json')
    saver = HelpSave()
    saver.init_save_dir(opt)
    train(opt, saver)


if __name__ == '__main__':
    main("VisionTransformer")
