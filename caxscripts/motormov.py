"""This class controls the CAX's motors for scanning its mirror."""

import time
# from typing import Literal
import numpy as np
from siriuspy.devices import CAXCtrl
from .h5file import HDF5File
from . import utils

from datetime import datetime

# from caxscripts import h5file
# import matplotlib.pyplot as plt

# from pyKinGalil_module import pyKinGalil as M1Kinematics

# path_mirror_gui_pkg = '/usr/local/scripts/gui/mirror-gui/kinUtils'
# sys.path.append(path_mirror_gui_pkg)

# ## Beamline parameters.
# Detector range.
ZPOSMIN = 135.14
ZPOSMAX = 560

SCAN_TYPES = {
    '1' : 'Mirror x translation (Tx)',
    '2' : 'Mirror x rotation    (Rx)',
    '3' : 'Mirror y translation (Ty)',
    '4' : 'Mirror y rotation    (Ry)',
    '5' : 'Mirror z rotation    (Rz)',
    '6' : 'Caustic',
    '7' : 'Lens (DVF2) translation',
    '8' : 'Slit set 1 window (before mirror)',
    '9' : 'Slit set 2 window (after mirror)',
}


def parameters_ask(scantype: str):
    """Get scan parameters from user."""
    st = SCAN_TYPES[scantype]
    scanset = dict()
    print(f"###\n###  {st} scan parameters\n###\n")
    scanset['start']  = float(input(
        f"Enter start {st} position/angle: "
        ))
    scanset['stop']   = float(input(
        f"Enter stop {st} position/angle: "
        ))
    scanset['nsteps'] = float(input(
        f"Enter number of steps for scanning: "
        ))
    scanset['dtime']  = float(input(
        "Enter dwell time at each position (seconds): "
        ))
    return scanset


def status_show(caxdev):
    """Show current status of the device."""
    print(f"\n##### {caxdev.devname} #####")
    status = caxdev.device_status()

    # DEBUG
    # print(f"\n>>>\n DEBUG: status dict: {status} \n<<<\n")
    # END DEBUG

    for key, val in status.items():
        print(f"\n -- {key.replace('_', ' ').title()}:\n")
        try:
            for pv, vs in val.items():
                print(f"  * {pv:15} : ", end="")
                for v in vs:
                    print(f"{v:.6f}, ", end="")
                print()
        except:
            for bl, v in val:
                print(f"  * {bl:15} : {v:10.4f}")
    print("\n#####\n")


class DeviceMove:
    """."""

    def __init__(self, scantype: int, caxdev, scanset=None):
        """."""
        self.caxdev  = caxdev
        self.cmov = caxdev.scan_function(scantype)
        self.scantype = scantype
        self.sct = SCAN_TYPES[self.scantype]
        self.scanset = scanset

    def device_scan(self):
        """."""
        try:
            status, err, results = self.scan_watch()
            if not status:
                raise Exception(
                    " ERROR : while executing "
                    f"{self.sct} scan:\n"
                    f" >>> {err}"
                    )
        except Exception as err:
            print(f" ERROR during {self.sct} scan:\n {err}")
            return None

        print(f" >>>>> {self.sct} scanning end time: {datetime.ctime(datetime.now())}\n")
        return results

    def scan_watch(self):
        """Status of movement."""
        print(f"\n\n >>>>> Starting {self.sct} scan... ")
        print(f" >>>>> Init time: {datetime.ctime(datetime.now())}\n")

        for pos in np.linspace(self.scanset['start'],
                               self.scanset['stop'],
                               self.scanset['nsteps']):
            print(f" Setting {self.sct} angle/position to {pos}... ", end="")
            try:
                results = self.cmov(pos)
            except Exception as err:
                return False, err, None
            time.sleep(self.scanset['dtime'])
            print(" done.")

        # Return: result status, error status, results
        return True, None, results


class CAXMirrorMove:
    """."""

    def __init__(self):
        """."""
        self.cax     = CAXCtrl()
        self.m1      = self.cax.mirror
        self.devname = 'CAX Mirror'

        # self.m1kin = m1kin
        # self._Motors = Literal[
        #     'ry', 'tx', 'y1', 'y2', 'y3', 'rx', 'rz', 'y'
        #     ]

        # WARNING: These are just place holders - methods must be implemented!

        # x coordinate.
        # self.tx_mov  = self.mirror_scan()
        # self.rx_mov  = self.mirror_scan()

        # y coordinate.
        # self.ty_mov  = self.mirror_scan()
        # self.ry_mov  = self.mirror_scan()

        # z coordinate.
        # self.rz_mov  = self.mirror_scan()

        # Lens scan.
        # self.lens    = self.lens_scan()

    def device_status(self):
        """Show initial statuts of mirror."""
        raw_motor_descr = {
            'TX' : 'x position',
            'RY' : 'y rotation'   ,
            'Y1' : 'leveler Z-',
            'Y2' : 'leveler X+',
            'Y3' : 'leveler Z+',
        }
        r_motor_list = list(raw_motor_descr.keys())
        #
        kin_motor_descr = {
            'CS_RX' : 'x rotation'   ,
            'CS_RY' : 'y rotation'   ,
            'CS_RZ' : 'z rotation'   ,
            'CS_TX' : 'x position',
            'CS_TY' : 'y position',
        }
        k_motor_list = list(kin_motor_descr.keys())

        raw_motor  = {
            raw_motor_descr[k] : [] for k in r_motor_list
            }
        kin_motor  = {
            kin_motor_descr[k] : [] for k in k_motor_list
            }

        dev_status = dict()
        count = 0

        # Get PVs.
        for key, val in vars(self.m1.PVS).items():
            pv = key.split('_')

            # Re-assemble PV name if kinematics.
            if pv[0] == 'CS':
                pv[0] = '_'.join(pv[:2])
                pv.pop(1)

            # Skip if PV name is not relevant for mirror status.
            if (len(pv) < 2 or
                pv[0] not in r_motor_list + k_motor_list or
                pv[1] not in ['MON', 'ENBL']):
                continue

            # Motor description and value.
            motval  = self.m1[val]

            # Select PVs: read back value and enabled status.
            if pv[1] in ['MON', 'ENBL']:
                # Select PVs: raw motors or kinematics.
                if pv[0] in r_motor_list:
                    motdesc = raw_motor_descr[pv[0]]
                    raw_motor[motdesc].append(motval)
                elif pv[0] in k_motor_list:
                    motdesc = kin_motor_descr[pv[0]]
                    kin_motor[motdesc].append(motval)

                # Show progress bar, for some checkings may be slow.
                count = self.progress_bar(self.devname, count, key)

        # Populate status dict.
        dev_status['raw_motor']  = raw_motor
        dev_status['kinematics'] = kin_motor

        # Done.
        print(f"\r Scanning {self.devname} status..."
              " done.            ", flush=True)

        return dev_status

    def progress_bar(self, devname, count, pv):
        """Show progress bar."""
        pb = ['|', '/', '-', '\\']
        print(f"\r Scanning {devname}  status ..."
              f" {pb[count % 4]} [{pv}]",
              end="", flush=True)
        return count + 1

    def device_scan(self, filename, filedir, motor, positions=None):
        """."""
        if positions is None:
            positions = self.mirror_positions(motor)

        file = HDF5File(filename, filedir)

        t0 = time.time()

        for i, pos in enumerate(positions):
            print(f"Step: {i+1}/{len(positions)} -"
                f" Moving mirror {motor} to position {pos}")
            self.set_mirror_pos(motor, pos)

            scaname = f'scan-{i:04d}'
            scanmetadata = {
                f'{motor}_pos'     : self.get_mirror_pos(motor),
                'photocollector'   : self.cax.mirror.photocurrent_signal,
            }

            dvf1img = utils.get_image(dvf=self.cax.dvf_A1)
            dvf1metadata = {
                'exposure_time'    : self.cax.dvf_A1.exposure_time,
                'acquisition_time' : self.cax.dvf_A1.acquisition_time
            }
            dvf2img = utils.get_image(dvf=self.cax.dvf_B1)
            dvf2metadata = {
                'exposure_time'    : self.cax.dvf_B1.exposure_time,
                'acquisition_time' : self.cax.dvf_B1.acquisition_time
            }

            file.save_group(grpname=scaname, grpmetadata=scanmetadata)
            file.save_dataset(grpname=scaname, dsetname='dvf1',
                            dsetmetadata=dvf1metadata,
                            dsetdata=dvf1img)
            file.save_dataset(grpname=scaname, dsetname='dvf2',
                            dsetmetadata=dvf2metadata,
                            dsetdata=dvf2img)

        t1 = time.time()

        print()
        print(f'Elapsed time [min]: {(t1-t0)/60:.2f}')

    """
    Dada a posicao (ry, tx, y1, y2, y3), a posição virtual (x, y, rx, ry, rz)
    é calculada em pkg_obj.direct_transf(*motorslist).

    motorlist vem de epics.caget_many(pvlist) e pvlist é do self.pvlist,
    onde self é da classe MirrorGui, então pvlist é [ry, tx, y1, y2, y3].

    pkg_obj é objeto pkg(kin_mode=0), onde pfg é classe pyKinGalil

    resumo da última:
    [tx, ty, rx, ry, rz] = pyKinGalil().direct_transf(ry, tx, y1, y2, y3)

    o inverso:
    [ry, tx, y1, y2, y3] = pyKinGalil().inverse_transf(tx, ty, rx, ry, rz)

    """

    def get_mirror_real(self):
        """."""
        ry, tx, y1, y2, y3 = (self.m1.ry_pos,
                              self.m1.tx_pos,
                              self.m1.y1_pos,
                              self.m1.y2_pos,
                              self.m1.y3_pos)
        return ry, tx, y1, y2, y3

    def set_mirror_real(self, ry, tx, y1, y2, y3):
        """."""
        self.m1.ry_pos = ry
        self.m1.tx_pos = tx
        self.m1.y1_pos = y1
        self.m1.y2_pos = y2
        self.m1.y3_pos = y3

    # def get_mirror_virtual(self):
    #     """."""
    #     ry, tx, y1, y2, y3 = self.get_mirror_real()
    #     tx, ty, rx, ry, rz = self.m1kin.direct_transf(ry, tx, y1, y2, y3)
    #     return tx, ty, rx, ry, rz

    # def set_mirror_virtual(self, tx, ty, rx, ry, rz):
    #     """."""
    #     ry, tx, y1, y2, y3 = self.m1kin.inverse_transf(tx, ty, rx, ry, rz)
    #     self.set_mirror_real(ry, tx, y1, y2, y3)

    def get_mirror_pos(self, motor):
        """."""
        if motor in ['ty', 'rx', 'rz']:
            idx_virtualmotor = {
                'ty' : 1,
                'rx' : 2,
                'rz' : 3
                }[motor]
            virtualposes = self.get_mirror_virtual()
            motor_pos = virtualposes[idx_virtualmotor]

        else:
            # ry or tx or y1 or y2 or y3
            motor_pos = getattr(self.m1, f'{motor}_pos')

        return motor_pos

    def set_mirror_pos(self, motor, pos):
        """Move specified real or virtual motor of CARCARA X Mirror.

        Parameters
        ----------
        motor : str, {'ry', 'tx', 'y1', 'y2', 'y3', 'rx', 'rz', 'y'}
            Real or virtual motor name.
        pos : float
            Motor position.
        """
        if motor in ['ty', 'rx', 'rz']:
            idx_virtualmotor = {
                'ty' : 1,
                'rx' : 2,
                'rz' : 3
                }[motor]
            virtualposes = self.get_mirror_virtual()
            virtualposes[idx_virtualmotor] = pos
            self.set_mirror_virtual(*virtualposes)

        else:
            # ry or tx or y1 or y2 or y3
            setattr(self.m1, f'{motor}_pos', pos)

        # !: waiting time after setting

    # ? what are all possibilities of save during scan?
    # mirror: motor position, photocollector signal
    # dvf1: image, exposure time, acquisition time
    # dvf2: image, exposure time, acquisition time

    def mirror_positions(self, n=100):
        """."""
        # real or virtual motor
        # np.linspace(pos_min, pos_max, N) for each motor
        raise NotImplementedError


# ----- Slits ------- #

class CAXSlitMove:
    """Class to control slit scan of CAX beamline."""

    def __init__(self, filename=None, filedir=None):
        """."""
        self.cax = CAXCtrl()
        self.devname = 'CAX Slit'

        self.h5file = (
            HDF5File(filename, filedir) if filename is not None else None
            )

    def device_status(self):
        """Show initial statuts of mirror."""
        slit_descr = {
            'TOP'    : 'top blade position',
            'BOTTOM' : 'bottom blade position',
            'LEFT'   : 'left blade position',
            'RIGHT'  : 'right blade position',
        }
        slit_list = list(slit_descr.keys())

        slits = list()
        dev_status = dict()

        # Get PVs.
        for key, val in vars(self.cax.slit_A1.PVS).items():
            pv = key.strip('CS').split('_')
            # Select PVs.
            if pv[0] in slit_list and pv[1] == 'MON':
                v = self.cax.slit_A1[val]
                slits.append((pv[0], v))
        dev_status['slits'] = slits
        return dev_status

    def set_slit_pos(self, side, pos):
        """."""
        side_func_dict = {
            'top'    : self.cax.slit_A1.move_robust_top,
            'bottom' : self.cax.slit_A1.move_robust_bottom,
            'left'   : self.cax.slit_A1.move_robust_left,
            'right'  : self.cax.slit_A1.move_robust_right
        }
        side_func = side_func_dict[side]
        side_func(value=pos)

    def set_slit_all(self, slit,
                     top_pos, bottom_pos,
                     left_pos, right_pos):
        """."""
        slit.move_robust_top(value=top_pos)
        slit.move_robust_bottom(value=bottom_pos)
        slit.move_robust_left(value=left_pos)
        slit.move_robust_right(value=right_pos)

    def slit_positions(self, m=100, n=100):
        """."""
        # 2 arrays of slit center positions: x and y
        raise NotImplementedError

    def scan_slit(self, slit, positions=None, sqsize=0.4):
        """."""
        if positions is None:
            xposes, yposes = self.slit_positions(slit)

        t0 = time.time()
        for i, posx in enumerate(xposes):
            for j, posy in enumerate(yposes):
                print(f"Step: Row {i+1}/{positions.shape[0]},\n"
                    f" Column {j+1}/{positions.shape[1]} -"
                    f" Moving slit {slit} (square size = {sqsize})\n"
                    f" to center position ({posx},{posy})")

                self.set_slit_all(slit,
                            posy+sqsize/2, posy-sqsize/2,
                            posx-sqsize/2, posx+sqsize/2)

                # Structured data saving.
                # Name, generic metadata.
                scaname = f'scan-{i:04d}-{j:04d}'
                scanmetadata = {
                    'slit_top'    : self.get_slit_pos(slit, 'top'),
                    'slit_bottom' : self.get_slit_pos(slit, 'bottom'),
                    'slit_left'   : self.get_slit_pos(slit, 'left'),
                    'slit_right'  : self.get_slit_pos(slit, 'right'),
                }

                # Data, metadata, save it.
                dvf1img = utils.get_image(dvf=self.cax.dvf_A1)
                dvf1metadata = {
                    'exposure_time'    : self.cax.dvf_A1.exposure_time,
                    'acquisition_time' : self.cax.dvf_A1.acquisition_time
                }
                dvf1metadata.update(scanmetadata)
                data_save(self.h5file, scaname, dvf1metadata,
                          dvf1img, devname='dvf1')

                # Data, metadata, save it.
                dvf2img = utils.get_image(dvf=self.cax.dvf_B1)
                dvf2metadata = {
                    'exposure_time'    : self.cax.dvf_B1.exposure_time,
                    'acquisition_time' : self.cax.dvf_B1.acquisition_time
                }
                dvf2metadata.update(scanmetadata)
                data_save(self.h5file, scaname, dvf2metadata,
                          dvf2img, devname='dvf2')

        t1 = time.time()
        print()
        print(f'Elapsed time [min]: {(t1-t0)/60:.2f}')


# ----- caustic ------- #

class CAXCausticMove:
    """Class to control caustic scan of CAX beamline."""

    def __init__(self, filename=None, filedir=None):
        """."""
        self.cax = CAXCtrl()
        self.devname = 'CAX Caustic'
        self.scan_function = self.caustic_scan
        self.caustic_scan_settings()

        self.h5file = (
            HDF5File(filename, filedir) if filename is not None else None
            )

    def caustic_scan_settings(self):
        """."""
        print("\n ** Settings")
        status_show(self)

        # Set start, stop, nsteps, dtime.
        caustic_settings = parameters_ask('6')
        self.start  = caustic_settings['start']
        self.stop   = caustic_settings['stop']
        self.nsteps = caustic_settings['nsteps']
        self.dtime  = caustic_settings['dtime']

        # Set scan positions.
        self.detector_positions_scan = np.linspace(
            self.start, self.stop, self.nsteps
            )

    def device_status(self):
        """Show initial statuts of caustic motor."""
        dev_status = {'caustic': [('Z', self.cax.dvf_B1.z_pos)]}
        return dev_status

    def set_detector_pos(self, pos):
        """."""
        self.cax.dvf_B1.z_pos = pos
        # !: waiting time after setting

    def get_detector_pos(self):
        """."""
        return self.cax.dvf_B1.z_pos

    def caustic_scan(self):
        """."""
        # saving beamline state before the scan
        utils.config_save(self.cax, self.h5file)

        # Get scanning positions.
        positions = self.detector_positions_scan

        t0 = time.time()
        for i, pos in enumerate(positions):
            print(f"Step: {i+1}/{len(positions)} -"
                  f" Moving detector to position {pos}")

            self.set_detector_pos(pos)
            self.scaname      = f'scan-{i:04d}'
            self.dvf2img      = utils.get_image(dvf=self.cax.dvf_B1)
            self.scanmetadata = {
                'z_pos': self.get_detector_pos(),
            }
            data_save(self.h5file, self.scaname, self.scanmetadata,
                      self.dvf2img, devname='dvf2')

        t1 = time.time()
        print()
        print(f'Elapsed time [min]: {(t1-t0)/60:.2f}')

    # def data_save(self):
    #     """."""
    #     if self.h5file:
    #         self.h5file.save_group(grpname=self.scaname,
    #                                grpmetadata=self.scanmetadata)
    #         self.h5file.save_dataset(grpname=self.scaname,
    #                                  dsetname='dvf2',
    #                                  dsetdata=self.dvf2img
    #                                  dsetmetadata=self.metadata
    #                                  )


# ------- lens -------- #

class CAXLensMove:
    """Class to control lens scan of CAX beamline."""

    def __init__(self, filename=None, filedir=None, positions_number=100):
        """."""
        self.cax = CAXCtrl()
        self.devname = 'CAX Lens'
        self.h5file = (
            HDF5File(filename, filedir) if filename is not None else None
            )
        self.position_number = positions_number

    def get_lens_pos(self):
        """."""
        return self.cax.dvf_B1.lens_pos

    def set_lens_pos(self, pos):
        """."""
        self.cax.dvf_B1.lens_pos = pos
        # !: waiting time after setting

    def device_scan(self, positions=None):
        """."""
        t0 = time.time()
        for i, pos in enumerate(positions):
            print(f"Step {i+1}/{self.position_number} -"
                f" Moving lens to position {pos}")
            self.set_lens_pos(pos)

            # Structured data saving.
            self.scaname     = f'scan-{i:04d}'
            self.dvf2img     = utils.get_image(dvf=self.cax.dvf_B1)
            self.setmetadata = {
                'lens_pos'         : self.get_lens_pos(),
                'exposure_time'    : self.cax.dvf_B1.exposure_time,
                'acquisition_time' : self.cax.dvf_B1.acquisition_time
            }
            data_save(self.h5file, self.scaname, self.setmetadata,
                      self.dvf2img, devname='dvf2')

        t1 = time.time()
        print()
        print(f'Elapsed time [min]: {(t1-t0)/60}')

    # def data_save(self):
    #     """."""
    #     self.h5file.save_group(grpname=self.scaname,
    #                            grpmetadata=self.scanmetadata)
    #     self.h5file.save_dataset(grpname=self.scaname,
    #                              dsetname='dvf2',
    #                              dsetmetadata=self.dvf2metadata,
    #                              dsetdata=self.dvf2img)


@staticmethod
def cax_current_config(cax: CAXCtrl):
    """."""
    current_config = dict()

    slit_1 = {
        'top'    : cax.slit_A1.top_pos,
        'bottom' : cax.slit_A1.bottom_pos,
        'left'   : cax.slit_A1.left_pos,
        'right'  : cax.slit_A1.right_pos
    }

    slit_2 = {
        'top'    : cax.slit_B1.top_pos,
        'bottom' : cax.slit_B1.bottom_pos,
        'left'   : cax.slit_B1.left_pos,
        'right'  : cax.slit_B1.right_pos
    }

    dvf_1 = {
        'acq_time'  : cax.dvf_A1.acquisition_time,
        'expo_time' : cax.dvf_A1.exposure_time
    }

    dvf_2 = {
        'acq_time'  : cax.dvf_A1.acquisition_time,
        'expo_time' : cax.dvf_B1.exposure_time,
        'z_pos'     : cax.dvf_B1.z_pos
    }

    mirror = {
        'ry'        : cax.mirror.ry_pos,
        'tx'        : cax.mirror.tx_pos,
        'y1'        : cax.mirror.y1_pos,
        'y2'        : cax.mirror.y2_pos,
        'y3'        : cax.mirror.y3_pos,
        'photocollector': cax.mirror.photocurrent_signal,
    }

    current_config['slit1']  = slit_1
    current_config['dvf1']   = dvf_1
    current_config['slit2']  = slit_2
    current_config['dvf2']   = dvf_2
    current_config['mirror'] = mirror

    return current_config


@staticmethod
def data_save(h5file, scaname, setmetadata, dvfimg, devname='dvf'):
    """."""
    if h5file:
        h5file.save_group(grpname=scaname)
        h5file.save_dataset(grpname=scaname,
                            dsetdata=dvfimg,
                            dsetname=devname,
                            dsetmetadata=setmetadata)


@staticmethod
def config_save(cax: CAXCtrl, h5file: HDF5File):
    """."""
    config = utils.cax_current_config(cax)
    h5file.save_group(grpname='beamline_config', grpmetadata=config)


# define a frame for slit where slit blade right and left
# move to same position

# if __name__ == "__main__":
#     """."""
#     cax = CAXCtrl()
#     m1kin = M1Kinematics()
#     caxmov = CAXMirrorMove(cax, m1kin)
