import os

class Installer:
    def __init__(self):
        self.env_name = None
        self.project_path = None
        self.teminal_cols = os.get_terminal_size().columns

    @staticmethod    
    def conda_always_yes(enable:bool = True) -> None:
        return
        if enable:
            os.system(f'mamba config --env --set always_yes true"')
        else:
            os.system('mamba config --env --remove-key always_yes')

    def new_env(self, activete:bool = True) -> None:
        name = self.env_name
        print(f"Creating new conda environment: {name}")
        os.system(f'mamba create -n {name} python=3.12 -y')
        if activete:
            os.system(f'mamba activate {name}')
        self.clear()
        
    def install_pytorch(self):
        self.header("Install Pytorch")
        print(f"Please follow the link bellow, select:" )
        print(f"PyTorch Build = stable")
        print(f"OS - the operating system of the computer [Windows/Linux]")
        print(f"Package - the package manager [conda]")
        print(f"Language - the programming language [Python]")
        print(f"Compute Platform - if the computer has GPU select the newest version of CUDA [12.1 was tested], otherwise select CPU")
        print("")
        print(f"Link: https://pytorch.org/get-started/locally/#start-locally")
        command = input(f"after selecting the options, copy the command and paste it here: ") + " -y"
        command = command.replace('conda ', 'mamba ', 1)
        print(f"Installing PyTorch...")
        os.system(command)
        
        print(f"Done")
        self.clear()

    def install_requirements(self):
        self.header(f"Installing requirements")
        os.system(f'mamba env update -f requirements.yaml -n {self.env_name}')
        self.clear()

    def remove_env(self):
        env_name = input("Enter the name of the environment to delete: ")
        os.system(f"mamba remove -n {env_name} --all -y")

    def list_env(self):
        print("here are the python environments installed on this computer: ")
        os.system("mamba env list")
        self.clear()

    def clear(self):
        print('-'*self.teminal_cols)

    def clean(self):
        self.header("Cleaning mamba")
        os.system("mamba clean -a -y")
    
    def header(self, name:str):
        if len(name) > 0:
            name = f" {name} "
        pad_size = max(0, (self.teminal_cols - len(name)) // 2)
        header = '#'*pad_size + name + "#"*pad_size
        print(header)

    def install_env(self):
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

    def main(self):
        self.header("Welcome to Project Worms")
        print(f"this is the installation script for the project")
        self.list_env()
        self.header("choose an action")
        print("i - install an environment")
        print("r - remove an evironment")
        print("c - clean mamba")
        action = input("action: ").lower()
        if action == 'i':
            self.install_env()
        elif action == 'r':
            self.remove_env()
        elif action == 'c':
            self.clean()
        else:
            print("ERROR:: unrecognized action, exiting..")
        



if __name__ == "__main__":
    installer = Installer()
    installer.main()












