import os

class Installer:
    def __init__(self):
        self.env_name = None
        self.project_path = None

    @staticmethod    
    def conda_always_yes(enable:bool = True) -> None:
        if enable:
            os.system(f'export CONDA_ALWAYS_YES="true"')
        else:
            os.system('unset CONDA_ALWAYS_YES')

    def new_env(self, activete:bool = True) -> None:
        name = self.env_name
        print(f"Creating new conda environment: {name}")
        os.system(f'mamba create -n {name} python=3.12 -y')
        if activete:
            os.system(f'mamba activate {name} -y')
        
    def install_pytorch(self):
        print(f"Please follow the link bellow, select: your OS [Windows/Linux], package = 'conda', " )
        print(f"PyTorch Build = stable")
        print(f"OS - the operating system of the computer [Windows/Linux]")
        print(f"Package - the package manager [conda]")
        print(f"Language - the programming language [Python]")
        print(f"Compute Platform - if the computer has GPU select the newest version of CUDA [12.1 was tested], otherwise select CPU")
        print("")
        print(f"Link: https://pytorch.org/get-started/locally/#start-locally")
        command = input(f"after selecting the options, copy the command and paste it here")
        command = command.replace('conda', 'mamba')
        print(f"Installing PyTorch...")
        os.system(command)
        
        print(f"Done")

    def install_requirements(self):
        print(f"Installing requirements...")
        os.system('mamba env update -f requirements.yaml')


    def run(self):
        print(f"Welcome to the installation script of the project")
        self.env_name = input(f"Enter the name of the new python environment: ")
        self.env_name = self.env_name.replace(' ', '_')
        self.project_path = input(f"Enter the path of the project folder: ")
        if not os.path.isdir(self.project_path):
            print(f"ERROR:: Invalid path, please verify the path and try again")
            return
        os.chdir(self.project_path)

        self.conda_always_yes(True)
        self.new_env()
        self.install_pytorch()
        self.install_requirements()
        self.conda_always_yes(False)
        
        print(f"Done")


if __name__ == "__main__":
    installer = Installer()
    installer.run()












