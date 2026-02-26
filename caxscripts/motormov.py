"""This class controls the CAX's motors for scanning its mirror."""

import time
import numpy as np
from . import utils


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


class CAXMirrorMove:
    """."""

    def __init__(self, caxctrl):
        """."""
        self.cax     = caxctrl
        self.m1      = self.cax.mirror
        self.devname = 'CAX Mirror'

    def device_status(self):
        """Show current status of mirror motors."""
        cax_status = utils.current_config(self.cax)
        mirror_status = cax_status['mirror']

        # mirror_status[key] = [mon, lolm, hilm, enbl]
        raw_motor_descr = {
            'tx': 'x position',
            'ry': 'y rotation',
            'y1': 'leveler Z-',
            'y2': 'leveler X+',
            'y3': 'leveler Z+',
        }
        kin_motor_descr = {
            'cs_rx': 'x rotation',
            'cs_rz': 'z rotation',
            'cs_tx': 'x position',
            'cs_ty': 'y position',
        }

        raw_motor = {
            desc: mirror_status[key]
            for key, desc in raw_motor_descr.items()
        }
        kin_motor = {
            desc: mirror_status[key]
            for key, desc in kin_motor_descr.items()
        }

        return {
            'raw_motor': raw_motor,
            'kinematics': kin_motor,
            'photocollector': mirror_status['photocollector']
            }

    def progress_bar(self, count, pv):
        """Show progress bar."""
        pb = ['|', '/', '-', '\\']
        print(f"\r Scanning {self.devname}  status ..."
              f" {pb[count % 4]} [{pv}]",
              end="", flush=True)
        return count + 1

    def device_scan(self, scanargs):
        """Scan a mirror motor and return collected data.

        Args:
            scanargs: dict with 'motor', 'start', 'stop', 'nsteps', 'dtime'.

        Returns:
            list of dicts, one per scan step.
        """
        motor     = scanargs['motor']
        positions = np.linspace(
            scanargs['start'], scanargs['stop'], int(scanargs['nsteps'])
        )
        results = []
        t0 = time.time()

        for i, pos in enumerate(positions):
            print(f"Step: {i+1}/{len(positions)} -"
                  f" Moving mirror {motor} to {pos}")
            self.m1.move(motor, pos)
            time.sleep(scanargs.get('dtime', 0))

            step = {'step': i, 'scan_type': 'mirror', 'scan_motor': motor}
            step.update(utils.snapshot_machine_state(self.cax))
            results.append(step)

        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')
        return results

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
        ry, tx, y1, y2, y3 = (self.m1.ry,
                              self.m1.tx,
                              self.m1.y1,
                              self.m1.y2,
                              self.m1.y3)
        return ry, tx, y1, y2, y3

    def set_mirror_real(self, ry, tx, y1, y2, y3):
        """."""
        self.m1.ry = ry
        self.m1.tx = tx
        self.m1.y1 = y1
        self.m1.y2 = y2
        self.m1.y3 = y3

    # def get_mirror_virtual(self):
    #     """."""
    #     ry, tx, y1, y2, y3 = self.get_mirror_real()
    #     tx, ty, rx, ry, rz = self.m1kin.direct_transf(ry, tx, y1, y2, y3)
    #     return tx, ty, rx, ry, rz

    # def set_mirror_virtual(self, tx, ty, rx, ry, rz):
    #     """."""
    #     ry, tx, y1, y2, y3 = self.m1kin.inverse_transf(tx, ty, rx, ry, rz)
    #     self.set_mirror_real(ry, tx, y1, y2, y3)

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

    def __init__(self, caxctrl):
        """."""
        self.cax = caxctrl
        self.devname = 'CAX Slit'

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

    def device_scan(self, scanargs):
        """Scan slit positions and return collected data.

        Args:
            scanargs: dict with 'slit' (device), 'xpositions', 'ypositions',
                      'sqsize', 'dtime'.

        Returns:
            list of dicts, one per grid point.
        """
        slit    = scanargs['slit']
        xposes  = scanargs['xpositions']
        yposes  = scanargs['ypositions']
        sqsize  = scanargs.get('sqsize', 0.4)
        results = []
        t0 = time.time()

        for i, posx in enumerate(xposes):
            for j, posy in enumerate(yposes):
                print(f"Step: Row {i+1}/{len(xposes)},"
                      f" Col {j+1}/{len(yposes)} -"
                      f" slit center ({posx:.3f}, {posy:.3f})")

                self.set_slit_all(slit,
                                  posy + sqsize/2, posy - sqsize/2,
                                  posx - sqsize/2, posx + sqsize/2)
                time.sleep(scanargs.get('dtime', 0))

                step = {'step_row': i, 'step_col': j, 'scan_type': 'slit'}
                step.update(utils.snapshot_machine_state(self.cax))
                results.append(step)

        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')
        return results


# ----- caustic ------- #

class CAXCausticMove:
    """Class to control caustic scan of CAX beamline."""

    def __init__(self, caxctrl):
        """."""
        self.cax     = caxctrl
        self.devname = 'CAX Caustic'

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

    def device_scan(self, scanargs):
        """Scan detector z position (caustic) and return collected data.

        Args:
            scanargs: dict with 'start', 'stop', 'nsteps', 'dtime'.

        Returns:
            list of dicts, one per scan step.
        """
        positions = np.linspace(
            scanargs['start'], scanargs['stop'], int(scanargs['nsteps'])
        )
        results = []
        t0 = time.time()

        for i, pos in enumerate(positions):
            print(f"Step: {i+1}/{len(positions)} -"
                  f" Moving detector to z={pos:.3f}")
            self.set_detector_pos(pos)
            time.sleep(scanargs.get('dtime', 0))

            step = {'step': i, 'scan_type': 'caustic'}
            step.update(utils.snapshot_machine_state(self.cax))
            results.append(step)

        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')
        return results


# ------- lens -------- #

class CAXLensMove:
    """Class to control lens scan of CAX beamline."""

    def __init__(self, caxctrl, positions_number=100):
        """."""
        self.cax     = caxctrl
        self.devname = 'CAX Lens'
        self.position_number = positions_number

    def get_lens_pos(self):
        """."""
        return self.cax.dvf_B1.lens_pos

    def set_lens_pos(self, pos):
        """."""
        self.cax.dvf_B1.lens_pos = pos
        # !: waiting time after setting

    def device_scan(self, scanargs):
        """Scan lens position and return collected data.

        Args:
            scanargs: dict with 'start', 'stop', 'nsteps', 'dtime'.

        Returns:
            list of dicts, one per scan step.
        """
        positions = np.linspace(
            scanargs['start'], scanargs['stop'], int(scanargs['nsteps'])
        )
        results = []
        t0 = time.time()

        for i, pos in enumerate(positions):
            print(f"Step {i+1}/{len(positions)} -"
                  f" Moving lens to {pos:.3f}")
            self.set_lens_pos(pos)
            time.sleep(scanargs.get('dtime', 0))

            step = {'step': i, 'scan_type': 'lens'}
            step.update(utils.snapshot_machine_state(self.cax))
            results.append(step)

        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')
        return results
