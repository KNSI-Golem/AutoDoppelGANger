from .data_preprocessor import DataPreprocessor
from .car_cutter import CarCutter
from nuimages import NuImages
from .gan import GAN
from .model_eval import ModelEval
from matplotlib import pyplot as plt
import torch
import json

class AutoDoppelGANgerShell:
    def __init__(self):
        print("Welcome to AutoDoppelGANger. Type 'help' to list commands.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GAN(64, 64, 3, 100, device, "../logs")
        self.model_eval = ModelEval(32, device)

    def help(self):
        print("Available commands:")
        print("lddst - Load dataset (Usage: lddst <filepath>)")
        print("gensam - Generate Samples (Usage: gensam <num_samples> <num_rows> <num_columns>)")
        print("ldwghts - Load model weights (Usage: ldwghts <path to discriminator weights> <path to generator weights>)")
        print("train - Start training the model (Usage: train <path to json file>)")
        print("incsc - Calculate inception score of the model")
        print("fid - Calculate FID of the model")
        print("exit - Exit the shell")

    def cut_nu_images(self, dataroot, out_path, version, min_size_x, min_size_y):
        print("Cutting out cars...")
        nuim = NuImages(dataroot=dataroot, version=version, verbose=False, lazy=True)
        cutter = CarCutter(nuim, min_size_x, min_size_y)
        cutter.cut_out_vehicles_from_dataset(dataroot, out_path+"car_cut/")
        print("Done.")

    def load_dataset(self, filepath, target_width=64, target_height=64, img_channels=3):
        print("Loading dataset...")
        try:
            preprocess = DataPreprocessor(target_width, target_height, img_channels)
            self.dataset = preprocess.load_dataset(filepath)
            print("Done.")
        except AttributeError as e:
            print(f"Attribute error: {e}")
        except ImportError as e:
            print(f"Import error: {e}")
        except TypeError as e:
            print(f"Type error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except RuntimeError as e:
            print(f"Runtime error: {e}")
        except OSError as e:
            print(f"OS error: {e}")

    def train(self, json_file):
        print("Training model...")
        try:
            with open(json_file, 'r') as file_handle:
                training_setup = json.load(file_handle)

            if not self.dataset:
                print("You must load dataset with 'lddst <filepath>' before training model.")
                return
            self.model.train(self.dataset, training_setup["num_epochs"], training_setup["batch_size"], training_setup["learning_rate"],
                            training_setup["beta1"], training_setup["beta2"], training_setup["time_limit"])
            if training_setup["save_weights"]:
                self.model.save_models_weigths(
                        "../models/checkpoints/discriminator.pth", "../models/checkpoints/generator.pth")
        except FileNotFoundError as e:
            print(f"Configuration file not found: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from configuration file: {e}")
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
        except AttributeError as e:
            print(f"Attribute error: {e}")
        except TypeError as e:
            print(f"Type error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        print("Done.")

    def generate_samples(self, num_samples, num_rows, num_cols):
        print("Displaying images...")
        try:
            num_samples = int(num_samples)
            num_rows = int(num_rows)
            num_cols = int(num_cols)
            images = self.model.generate_samples(num_samples)
            _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i].permute(1, 2, 0).clamp(0, 1))
                ax.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
            print("Done...")
        except ValueError:
            print("All arguments must be integers: <num_samples> <num_rows> <num_columns>")

    def load_weights(self, path_dsc, path_gen):
        try:
            self.model.load_model_weights(path_dsc, path_gen)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except (torch.serialization.pickle.UnpicklingError, EOFError) as e:
            print(f"Unpickling error: {e}")
        except RuntimeError as e:
            print(f"Runtime error: {e}")

    def display_inception_score(self):
        print("Calculating Inception Score...")
        generated_images = self.model.generate_samples(1024)
        inc_score = self.model_eval.compute_inception_score(generated_images)
        print(f"Inception score of GAN model: {inc_score:.4f}")

    def display_FID(self):
        print("Calculating FID...")
        generated_images = self.model.generate_samples(1024)
        FID_score = self.model_eval.compute_FID(self.model.load_data, generated_images)
        print(f"Inception score of GAN model: {FID_score:.4f}")

    def exit(self):
        print("Exiting...")
        return False

    def run(self):
        running = True
        while running:
            user_input = input("AutoDoppelGANger> ").strip()
            if user_input:
                parts = user_input.split()
                if parts[0] == 'help':
                    self.help()
                elif parts[0] == 'lddst':
                    try:
                        self.load_dataset(parts[1])
                    except IndexError:
                        print("Usage: lddst <filepath>")
                elif parts[0] == 'train':
                    try:
                        self.train(parts[1])
                    except IndexError:
                        print("Usage: train <path to json file>")
                elif parts[0] == 'ldwghts':
                    try:
                        self.load_weights(parts[1], parts[2])
                    except IndexError:
                        print("Usage: ldwghts <path to discriminator weights> <path to generator weights>")
                elif parts[0] == 'gensam':
                    try:
                        self.generate_samples(parts[1], parts[2], parts[3])
                    except IndexError:
                        print("Usage: gensam <num_samples> <num_rows> <num_columns>")
                elif parts[0] == 'fid':
                    self.display_FID()
                elif parts[0] == 'incsc':
                    self.display_inception_score()
                elif parts[0] == 'exit':
                    running = self.exit()
                else:
                    print("Unknown command. Type 'help' for help.")
            else:
                continue

