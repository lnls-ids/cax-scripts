"""This class controls the CAX's motors for scanning its mirror."""

from datetime import datetime
import time
import numpy as np
from . import utils
import matplotlib.pyplot as plt

from siriuspy.devices.device import _PVAccessor

# from caxscripts import h5file
# import matplotlib.pyplot as plt

# from pyKinGalil_module import pyKinGalil as M1Kinematics

# path_mirror_gui_pkg = '/usr/local/scripts/gui/mirror-gui/kinUtils'
# sys.path.append(path_mirror_gui_pkg)

# ## Beamline parameters.

# Detector range.
ZPOSMIN = 135.14
ZPOSMAX = 560


class CAXMirrorMove:
    """."""

    def __init__(self, caxctrl):
        """."""
        self.cax     = caxctrl
        self.m1      = self.cax.mirror
        self.devname = 'CAX Mirror'
        self.results = None

    def device_status(self):
        """Show current status of mirror motors."""
        cax_status = utils.snapshot_machine_state(self.cax)
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

    def _step_update_and_save(self, h5file, step_index, motor, step_type):
        """Update the step dict with current machine state and save to HDF5 file."""
        step = {
            'step': step_index,
            'scan_type': 'mirror',
            'scan_motor': motor,
            'state': step_type,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        step.update(utils.snapshot_machine_state(self.cax))
        self.results.append(step)
        utils.save_step(h5file, step)

        return step

    def device_scan(self, scanargs, h5file=None):
        """Scan a mirror motor and return collected data.

        Args:
            scanargs: dict with 'motor', 'start', 'stop', 'nsteps', 'dtime'.
            h5file: HDF5File instance for per-step saving (None to skip).

        Returns:
            list of dicts, one per scan step.
        """
        motor     = scanargs['motor']
        positions = np.linspace(
            scanargs['start'], scanargs['stop'], int(scanargs['nsteps'])
        )

        self.results = []
        t0 = time.time()

        # Record initial machine state before any movement.
        step0 = self._step_update_and_save(h5file, 0, motor,
                                          step_type='scanning')  

        for idx, pos in enumerate(positions):
            print(f"\n Step: {idx+1}/{len(positions)} -"
                  f" Moving mirror {motor} to {pos}")
            self.m1.move(motor, pos)
            time.sleep(scanargs.get('dtime', 0))

            try:
                self._step_update_and_save(h5file, idx+1, motor,
                                           step_type='scanning')
                print(f" Ended step {idx+1} -- snapshot saved. \n")
            except Exception as err:
                print(f" Could not save step {idx+1}: \n {err}\n")
                continue

        # Return motor to initial position.
        self.m1.move(motor, step0['mirror'][motor][0])
        
        try:
            self._step_update_and_save(h5file, len(positions), motor,
                                       step_type='final')
            print(" Ended final step -- snapshot saved. \n")
        except Exception as err:
            print(f" Could not save final step: \n {err}\n")
        
        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')

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

    def __init__(self, caxctrl, devname='slit_A1'):
        """."""
        self.cax        = caxctrl
        self.devname    = devname
        self.device     = getattr(self.cax, devname)
        self.dev_status = utils.snapshot_machine_state(self.cax)
        self.slit_image = self.dev_status['dvf_A1']['image']
        self.results    = None

    def device_status(self):
        """Show initial statuts of mirror."""
        slit_descr = {
            'TOP'    : 'top blade position',
            'BOTTOM' : 'bottom blade position',
            'LEFT'   : 'left blade position',
            'RIGHT'  : 'right blade position',
        }
        self.dev_status = utils.snapshot_machine_state(self.cax)
        for slit, vals in self.dev_status[self.devname].items():
            print(f" {slit.upper():<10}: {vals[0]:10.4f}"
                  f" [{vals[1]:10.4f}, {vals[2]:10.4f}]"
                  f" ({'enabled' if vals[3] else 'disabled'})")
        return self.dev_status

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

    # # The methods below convert pixel intervals observed by the DVF
    # # into actual motor positions.
    # def _top_pos_from_pixels(self, pixel):
    #     return -pixel * 0.01039 + 19.2

    # def _bottom_pos_from_pixels(self, pixel):
    #     return (pixel - 460) * 0.01041 + 37.3

    # def _left_pos_from_pixels(self, pixel):
    #     return -pixel * 0.007126 + 44.8

    # def _right_pos_from_pixels(self, pixel):
    #     return (pixel - 174) * 0.007126 + 45.9

    def set_slit_all(self,
                     top_pos, bottom_pos,
                     left_pos, right_pos):
        """."""

        # Calls robust_device_motor_move from _PVAccessor
        self.device.top    = top_pos
        self.device.bottom = bottom_pos
        self.device.left   = left_pos
        self.device.right  = right_pos

    def set_slit_all_from_pixels(self,
                     top, bottom, left, right):
        """Set positions of all slits calculated from image pixels."""
        # Calls robust_device_motor_move from _PVAccessor
        self.device.top    = self._top_pos_from_pixels(top)
        self.device.bottom = self._bottom_pos_from_pixels(bottom)
        self.device.left   = self._left_pos_from_pixels(left)
        self.device.right  = self._right_pos_from_pixels(right)

    def slit_positions(self, m=100, n=100):
        """."""
        # 2 arrays of slit center positions: x and y
        raise NotImplementedError

    def _step_update_and_save(self, h5file, slit, step_index, window, step_type):
        """Update the step dict with current machine state and save to HDF5 file."""
        (i, j, winsize) = window
        step = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'scan_type': slit,
            'step': step_index,
            'step_row': i,
            'step_col': j,
            'window_size' : winsize,
            'state': step_type,
        }
        step.update(utils.snapshot_machine_state(self.cax))
        self.results.append(step)
        utils.save_step(h5file, step)

        return step
    
    def device_scan(self, scanargs, h5file=None):
        """Scan slit positions and return collected data.

        Args:
            scanargs: dict with 'slit' (device), 'xpositions', 'ypositions',
                      'sqsize', 'dtime'.
            h5file: HDF5File instance for per-step saving (None to skip).

        Returns:
            list of dicts, one per grid point.
        """
        slit    = scanargs['slit']
        top     = scanargs['top_pos']
        bottom  = scanargs['bottom_pos']
        left    = scanargs['left_pos']
        right   = scanargs['right_pos']
        winsize = scanargs['winsize']

        self.results = []
        t0 = time.time()

        # Record initial machine state before any movement.
        self._step_update_and_save(h5file, slit, 0,
                                   (None, None, winsize),
                                   step_type='initial')
                                           
        step_idx = 1

        for j, topp in enumerate(top):
           for i, leftp in enumerate(left):
                print(f"Step: Row {i+1}/{len(left)},"
                      f" Col {j+1}/{len(top)} -"
                      f" slit center ({leftp:.3f}, {topp:.3f})")

                self.set_slit_all(topp, bottom[j], leftp, right[i])
                time.sleep(scanargs.get('dtime', 0))

                try:
                    self._step_update_and_save(h5file, slit, step_idx,
                                               (i, j, winsize),
                                               step_type='scanning')
                    print(f" Ended step Row {i+1}, Col {j+1} -- snapshot saved. \n")
                except Exception as err:
                    print(f" Could not save step Row {i+1}, Col {j+1}: \n {err}\n")
                    continue
                step_idx += 1

        # Return to initial slit positions after scan.
        self.set_slit_all(*scanargs['slit_initial_status'])
        try:
            self._step_update_and_save(h5file, slit, step_idx,
                                       (None, None, winsize),
                                       step_type='final')
            print(" Ended final step -- snapshot saved. \n")
        except Exception as err:
            print(f" Could not save final step: \n {err}\n")

        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')



# ----- caustic ------- #

class CAXCausticMove():
    """Class to control caustic scan of CAX beamline."""

    def __init__(self, caxctrl):
        """."""
        self.cax     = caxctrl
        self.devname = 'CAX Caustic'
        self.results  = None

    def device_status(self):
        """Show initial statuts of caustic motor."""
        dev_status = {'caustic': [('Z', self.cax.dvf_B1.z_mon)]}
        return dev_status

    def set_detector_pos(self, pos):
        """."""
        self.cax.dvf_B1.z = pos # this now uses _PVAccessor's robust moving methods
        # !: waiting time after setting

    def get_detector_pos(self):
        """."""
        return self.cax.dvf_B1.z_mon

    def _step_update_and_save(self, h5file, step_index, step_type):
        """Update the step dict with current machine state and save to HDF5 file."""
        step = {
            'step': step_index,
            'scan_type': 'caustic',
            'state': step_type,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        step.update(utils.snapshot_machine_state(self.cax))
        utils.save_step(h5file, plt.step)
        self.results.append(step)

        return step

    def device_scan(self, scanargs, h5file=None):
        """Scan detector z position (caustic) and return collected data.

        Args:
            scanargs: dict with 'start', 'stop', 'nsteps', 'dtime'.
            h5file: HDF5File instance for per-step saving (None to skip).

        Returns:
            list of dicts, one per scan step.
        """
        # List for storing results of each step, to be returned at the end.
        self.results = []
        t0 = time.time()

        # Record initial machine state before any movement.
        step0 = self._step_update_and_save(h5file, 0,
                                           step_type='initial')

        # Scan positions.
        positions = np.linspace(
            scanargs['start'], scanargs['stop'], int(scanargs['nsteps'])
        )

        for idx, pos in enumerate(positions):
            print(f"Step: {idx+1}/{len(positions)} -"
                  f" Moving detector to z = {pos:.3f}")
            self.set_detector_pos(pos)
            time.sleep(scanargs.get('dtime', 0))

            try:
                self._step_update_and_save(h5file, idx+1,
                                           step_type='scanning')
                print(f" Ended step {idx+1} -- snapshot saved. \n")
            except Exception as err:
                print(f" Could not save step {idx+1}: \n {err}\n")
                continue

        # Return to initial position after scan.
        self.set_detector_pos(step0['dvf_B1']['z_pos'][0])
        try:
            self._step_update_and_save(h5file, len(positions),
                                       step_type='final')
            print(" Ended final step -- snapshot saved. \n")
        except Exception as err:
            print(f" Could not save final step: \n {err}\n")

        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')


# ------- lens -------- #

class CAXLensMove(_PVAccessor):
    """Class to control lens scan of CAX beamline."""

    def __init__(self, caxctrl, positions_number=100):
        """."""
        self.cax             = caxctrl
        self.devname         = 'CAX Lens'
        self.position_number = positions_number
        self.results         = None

    def get_lens_pos(self):
        """."""
        return self.cax.dvf_B1.lens_mon

    def set_lens_pos(self, pos):
        """."""
        self.cax.dvf_B1.lens = pos  # this now uses _PVAccessor's robust moving methods
        # !: waiting time after setting

    def _step_update_and_save(self, h5file, step_index, step_type):
        """Update the step dict with current machine state and save to HDF5 file."""
        step = {
            'step': step_index,
            'scan_type': 'lens',
            'state': step_type,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        step.update(utils.snapshot_machine_state(self.cax))        
        utils.save_step(h5file, plt.step)
        self.results.append(step)

        return step

    def device_scan(self, scanargs, h5file=None):
        """Scan lens position and return collected data.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        Args:
            scanargs: dict with 'start', 'stop', 'nsteps', 'dtime'.
            h5file: HDF5File instance for per-step saving (None to skip).

        Returns:
            list of dicts, one per scan step.
        """
        positions = np.linspace(
            scanargs['start'], scanargs['stop'], int(scanargs['nsteps'])
        )
        self.results = []
        t0 = time.time()

        # Record initial machine state before any movement.
        step0 = self._step_update_and_save(h5file, 0, 
                                           step_type='initial')

        for idx, pos in enumerate(positions):
            print(f"Step {idx+1}/{len(positions)} -"
                  f" Moving lens to {pos:.3f}")
            self.set_lens_pos(pos)
            time.sleep(scanargs.get('dtime', 0))

            try:
                self._step_update_and_save(h5file, idx+1,
                                           step_type='scanning')
                print(f" Ended step {idx+1} -- snapshot saved. \n")
            except Exception as err:
                print(f" Could not save step {idx+1}: \n {err}\n")
                continue
        
        # Return to initial position after scan.
        self.set_lens_pos(step0['dvf_B1']['lens_pos'][0])
        try:
            self._step_update_and_save(h5file, len(positions),
                                       step_type='final')
            print(" Ended final step -- snapshot saved. \n")
        except Exception as err:
            print(f" Could not save final step: \n {err}\n")

        elapsed = (time.time() - t0) / 60
        print(f'\nElapsed time [min]: {elapsed:.2f}')


IMG_THRESHOLD = 100


