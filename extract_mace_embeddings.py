'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
'''

import torch
from mace import data
from mace.tools import torch_geometric, utils
from mace.calculators import mace_mp

emb_size_map = {"large": 1024, "medium": 512, "medium-0b": 512, "small": 128, "small-0b": 128}

class InteractionHead():
    def __init__(self, model="small", device="cuda", default_type="float32", checkpoint=None):
        '''
        model (str, optional): Path to the model. Defaults to "large". First checks for
            a local model and then downloads the default model from figshare. Specify "small",
            "medium" or "large" to download a smaller or larger model from figshare.
        device (str, optional): Device to use for the model. Defaults to "cuda".
            Use "cpu" for jupyter notebooks, "cuda" for scripts.
        default_dtype (str, optional): Default dtype for the model. Defaults to "float32".
            "float32" recommended for MD, "float64" recommended for geometry optimization
        '''
        self.model_type = model
        self.emb_size = emb_size_map[self.model_type]
        self.device = device
        self.checkpoint = checkpoint
        self.charges_key = "Qs"
        
        # load the model
        self.calc = mace_mp(model=self.model_type, default_dtype=default_type, device=self.device)
        
        # load finetuned weights from model checkpoint
        if self.checkpoint != None:
            self.calc.model.load_state_dict({k:v for k,v in torch.load(self.checkpoint, map_location=self.device)['state_dict'].items()})
        

        self.r_max = self.calc.models[0].r_max.item()
        self.z_table = z_table = utils.AtomicNumberTable([int(z) for z in self.calc.models[0].atomic_numbers])

    def _atoms_to_batch(self, atoms):
        # Handing for different versions of MACE
        try:
            config = data.config_from_atoms(atoms)
        except:
            config = data.config_from_atoms(atoms, charges_key=self.charges_key)

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        return batch

    def forward(self, atoms):
        # format atoms
        batch = self._atoms_to_batch(atoms)
        self.output = self.calc.models[0](batch)
        return self.output

    def calculate(self, atoms):
        # Use like ASE calculator
        atoms.calc = self.calc
        self.results = {"energy": atoms.get_potential_energy(), "forces": atoms.get_forces()}
        return self.results




