"""This class controls the CAX's motors for scanning its mirror."""

import sys
import time
from typing import Literal
import numpy as np
from siriuspy.devices import CAXCtrl
from .h5file import HDF5File
from . import utils
from pyKinGalil_module import pyKinGalil as M1Kinematics

path_mirror_gui_pkg = '/usr/local/scripts/gui/mirror-gui/kinUtils'
sys.path.append(path_mirror_gui_pkg)

# ## Beamline parameters.
# Detector range.
ZPOSMIN = 135.14
ZPOSMAX = 560


class CAXMirrorMove:
    """."""

    def __init__(self, cax: CAXCtrl, m1kin: M1Kinematics):
        """."""
        self.cax   = cax
        self.m1    = cax.mirror
        self.m1kin = m1kin
        self._Motors = Literal['ry', 'tx', 'y1', 'y2', 'y3', 'rx', 'rz', 'y']

        self.tx_mov  = self.scan_mirror()
        self.rx_mov  = self.scan_mirror()
        self.ty_mov  = self.scan_mirror()
        self.ry_mov  = self.scan_mirror()
        self.rz_mov  = self.scan_mirror()
        self.caustic = self.scan_mirror()
        self.lens    = self.scan_mirror()

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

    # todo: check if M1Kin(kin_mode=0) or M1Kin(kin_mode=1) change something.
    # I expect not

    def virtual_motor():
        """."""
        # /usr/local/scripts/gui/mirror-gui
        # x, y, rx, ry, rz  ->  ry, tx, y1, y2, y3
        # use the method inverse_transf of the pyKinGalil class
        # G = Matrix([-.X_sirius+X_newton, -.Y_sirius+Y_newton, -.Rx+Rx_newton,
        # -.Ry+Ry_newton, -.Rz+Rz_newton])
        # base_info is a class variable, a SimpleNamespace(),
        # filled along the methods of the class
        # .setup() fills base_info

        # base_info
        # .X_sirius, .Y_sirius, .Z_sirius = .XYZ
        # .XYZ = .T * Matrix([[0], [0], [0], [1]])
        # .T = .Ta0 * .T01 * .T12 * .T23 * .T34 * .T45 * .T56 * .T67 * .T78
        # .Ta0 = 1
        # .T01 = HomTransl_y(-.dist1-.dist2)
        # .T12 = HomTransl_x(.q2)
        # .T23 = HomRot_y(.Ry_pure)
        # .T34 = HomTransl_y(.dist1)
        # .T45 = HomTransl_y(.q1)
        # .T56 = HomRot_x(.Rx_pure)
        # .T67 = HomRot_z(.Rz_pure)
        # .T78 = HomTransl_y(.dist2)
        # .dist1 = symbol('Loc_dist1')
        # .dist2 = symbol('Loc_dist2')
        # .q2 = .Ux * .base_resolution
        # .Ux = .Motor_B
        # .base_resolution = symbol('Loc_base_res')

        # .Ry_pure = (0.369287740067191 - 0.369287740067191 *
        #             sqrt(1 - 0.0294338642691296 * .Uz * .base_resolution))

        # .Uz = .Motor_A
        # .q1 = .Uy_lev
        # .Uy_lev = symplify(.solutions[0][2])
        # .solutions = solve(equations, (.KinPosAxisA, .KinPosAxisC, .Uy))

        # equations = [
        #   series(series(.r1_0[1]/.leveler_enc_res-.KinPosMotor1,
        #                 x=.KinPosAxisA, x0=0, n=3, dir='+').removeO(),
        #                 x=.KinPosAxisC, x0=0, n=3, dir='+').removeO(),
        #   series(series(.r2_0[1]/.leveler_enc_res-.KinPosMotor2,
        #                 x=.KinPosAxisA, x0=0, n=3, dir='+').removeO(),
        #                 x=.KinPosAxisC, x0=0, n=3, dir='+').removeO(),
        #   series(series(.r3_0[1]/.leveler_enc_res-.KinPosMotor3,
        #                 x=.KinPosAxisA, x0=0, n=3, dir='+').removeO(),
        #                 x=.KinPosAxisC, x0=0, n=3, dir='+').removeO()
        # ]

        # .r1_0 = (Rot_x(.KinPosAxisA) * Rot_z(.KinPosAxisC) * .r1 +
        #          .Uy * np.array([[0], [1], [0]]))
        # .r2_0 = (Rot_x(.KinPosAxisA) * Rot_z(.KinPosAxisC) * .r2 +
        #          .Uy*np.array([[0], [1], [0]]))
        # .r3_0 = (Rot_x(.KinPosAxisA) * Rot_z(.KinPosAxisC) * .r3 +
        #          .Uy * np.array([[0], [1], [0]]))

        # .KinPosAxisA = symbol('KinPosAxisA')
        # .KinPosAxisC = symbol('KinPosAxisC')
        # .r1 = np.array([[.x1], [.y1], [.z1]])
        # .r2 = np.array([[.x2], [.y2], [.z2]])
        # .r3 = np.array([[.x3], [.y3], [.z3]])
        # .x1, .y1, .z1 = symbol('Loc_x1'), 0, symbol('Loc_z1')
        # .x2, .y2, .z2 = symbol('Loc_x2'), 0, symbol('Loc_z2')
        # .x3, .y3, .z3 = symbol('Loc_x3'), 0, symbol('Loc_z3')
        # .Uy = symbol('Uy')
        # .leveler_enc_res = symbol('Loc_leveler_res')
        # .KinPosMotor1 = .Motor_C
        # .Rx_pure = simplify(.solutions[0][0])
        # .Rz_pure = simplify(.solutions[0][1])
        # .Rx = .Rx_pure
        # .Ry = .Ry_pure
        # .Rz = .Rz_pure
        pass

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

    def get_mirror_virtual(self):
        """."""
        ry, tx, y1, y2, y3 = self.get_mirror_real()
        tx, ty, rx, ry, rz = self.m1kin.direct_transf(ry, tx, y1, y2, y3)
        return tx, ty, rx, ry, rz

    def set_mirror_virtual(self, tx, ty, rx, ry, rz):
        """."""
        ry, tx, y1, y2, y3 = self.m1kin.inverse_transf(tx, ty, rx, ry, rz)
        self.set_mirror_real(ry, tx, y1, y2, y3)

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

    def scan_mirror(self, filename, filedir, motor, positions=None):
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

    # ----- Slits ------- #

    def get_slit_pos(self, slit, side):
        """."""
        return getattr(slit, f'{side}_pos')

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

    def set_slit_all(self, slit, top_pos, bottom_pos, left_pos, right_pos):
        """."""
        slit.move_robust_top(value=top_pos)
        slit.move_robust_bottom(value=bottom_pos)
        slit.move_robust_left(value=left_pos)
        slit.move_robust_right(value=right_pos)

    def slit_positions(self, m=100, n=100):
        """."""
        # 2 arrays of slit center positions: x and y
        raise NotImplementedError

    def scan_slit(self, filename, filedir, slit, positions=None, sqsize=0.4):
        """."""
        if positions is None:
            xposes, yposes = self.slit_positions(slit)

        file = HDF5File(filename, filedir)
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

                scaname = f'scan-{i:04d}-{j:04d}'
                scanmetadata = {
                    'slit_top': self.get_slit_pos(slit, 'top'),
                    'slit_bottom': self.get_slit_pos(slit, 'bottom'),
                    'slit_left': self.get_slit_pos(slit, 'left'),
                    'slit_right': self.get_slit_pos(slit, 'right'),
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
                                dsetmetadata=dvf1metadata, dsetdata=dvf1img)
                file.save_dataset(grpname=scaname, dsetname='dvf2',
                                dsetmetadata=dvf2metadata, dsetdata=dvf2img)

        t1 = time.time()
        print()
        print(f'Elapsed time [min]: {(t1-t0)/60:.2f}')

    # ----- caustic ------- #

    def set_detector_pos(self, pos):
        """."""
        self.cax.dvf_B1.z_pos = pos
        # !: waiting time after setting

    def get_detector_pos(self):
        """."""
        return self.cax.dvf_B1.z_pos

    def detector_positions(self, n=100):
        """."""
        return np.linspace(ZPOSMAX, ZPOSMIN, n)

    def caustic_scan(self, filename, filedir, positions=None):
        """."""
        # saving beamline state before the scan
        utils.save_beamline_config(filename, filedir)

        if positions is None:
            positions = self.detector_positions()

        h5file = HDF5File(filename, filedir)
        t0 = time.time()
        for i, pos in enumerate(positions):
            print(f"Step: {i+1}/{len(positions)} -"
                f" Moving detector to position {pos}")

            self.set_detector_pos(pos)
            scaname = f'scan-{i:04d}'
            scanmetadata = {
                'z_pos': self.get_detector_pos(),
            }

            dvf2img = utils.get_image(dvf=self.cax.dvf_B1)
            h5file.save_group(grpname=scaname, grpmetadata=scanmetadata)
            h5file.save_dataset(grpname=scaname, dsetname='dvf2',
                                dsetdata=dvf2img)

        t1 = time.time()
        print()
        print(f'Elapsed time [min]: {(t1-t0)/60:.2f}')

    # ------- lens -------- #

    def get_lens_pos(self):
        """."""
        return self.cax.dvf_B1.lens_pos

    def set_lens_pos(self, pos):
        """."""
        self.cax.dvf_B1.lens_pos = pos
        # !: waiting time after setting

    def lens_positions(self, n=100):
        """."""
        return n

    def scan_lens(self, filename, filedir, positions=None):
        """."""
        if positions is None:
            positions = self.lens_positions()

        file = HDF5File(filename, filedir)
        t0 = time.time()

        for i, pos in enumerate(positions):
            print(f"Step {i+1}/{len(positions)} -"
                f" Moving lens to position {pos}")
            self.set_lens_pos(pos)
            scaname = f'scan-{i:04d}'
            scanmetadata = {
                'lens_pos': self.get_lens_pos(),
            }

            dvf2img = utils.get_image(dvf=self.cax.dvf_B1)
            dvf2metadata = {
                'exposure_time'    : self.cax.dvf_B1.exposure_time,
                'acquisition_time' : self.cax.dvf_B1.acquisition_time
            }

            file.save_group(grpname=scaname, grpmetadata=scanmetadata)
            file.save_dataset(grpname=scaname, dsetname='dvf2',
                            dsetmetadata=dvf2metadata, dsetdata=dvf2img)

        t1 = time.time()
        print()
        print(f'Elapsed time [min]: {(t1-t0)/60}')


# define a frame for slit where slit blade right and left
# move to same position

# if __name__ == "__main__":
#     """."""
#     cax = CAXCtrl()
#     m1kin = M1Kinematics()
#     caxmov = CAXMirrorMove(cax, m1kin)
