"""Parameters and constants for Carcara scripts."""

from siriuspy.devices import CAXCtrl


class Config:
    """Configuration parameters and constants for Carcara scripts."""

    cax = CAXCtrl()
    CAX_PREFIX = f"{cax.devname}:"

    # dvf
    SCALE = 1     # [um/px]
    MAXERRORCOUNT = 3

    IMG_THRESHOLD = 100

    # ry
    STEP  = 0.0001  # [mm]
    DELAY = 2      # [s]

    # slit1 limits
    TOPMIN   = 15
    TOPMAX   = 19.8
    #
    RIGHTMIN = 44.88
    RIGHTMAX = 45.9
    #
    BOTTOMIN = 31
    BOTTOMAX = 35.8
    #
    LEFTMIN  = 43.55
    LEFTMAX  = 44.56
    #
    TOPMID   = (TOPMIN   + TOPMAX)   / 2
    RIGHTMID = (RIGHTMIN + RIGHTMAX) / 2
    BOTTOMID = (BOTTOMIN + BOTTOMAX) / 2
    LEFTMID  = (LEFTMIN  + LEFTMAX)  / 2

    # Keys inside a step dict whose values are DVF snapshot dicts
    # (contain images).
    DVF_KEYS = ('dvf_A1', 'dvf_B1')

    # PV for storage ring current.
    SRPV = {'Storage ring current': "SI-13C4:DI-DCCT:Current-Mon"}

    # PVs for ID's gaps and phases.
    IDPVS = {
        "ARIRANHA ID phase" : "SI-20SB:ID-APU22:Phase-Mon",
        "CARNAUBA ID gap"   : "SI-06SB:ID-VPU29:KParam-Mon",
        "CATERETE ID gap"   : "SI-07SP:ID-VPU29:KParam-Mon",
        "EMA ID gap"        : "SI-08SB:ID-IVU18:KParam-Mon",
        "IPE ID gap"        : "SI-11SP:ID-UE44:KParam-Mon",
        "MANACA ID phase"   : "SI-09SA:ID-APU22:Phase-Mon",
        "PAINEIRA ID gap"   : "SI-14SB:ID-IVU18:KParam-Mon",
        "SABIA ID gap"      : "SI-10SB:ID-DELTA52:KParam-Mon",
        "SAPUCAIA ID phase" : "SI-17SA:ID-APU22:Phase-Mon",
    }

    # Slit limits defined from DVF images with slits fully open.
    # This avoids long motor moves and discrepancies between
    # hard limits and those defined in the front end.
    SLIT_A1_TOP_LIMS = {
        'TOP'    : 19.5,
        'BOTTOM' : 37.5,
        'LEFT'   : 47.5,
        'RIGHT'  : 46.5,
    }

    # Environment PVs for the CAX mirror.
    # Water flux.
    PVFLUX = [
        CAX_PREFIX + cax.mirror.PVS.FM_01_MON,
        CAX_PREFIX + cax.mirror.PVS.FM_02_MON,
    ]

    # Temperature.
    PVTEMP = [
        CAX_PREFIX + cax.mirror.PVS.TP_00_MON,
        CAX_PREFIX + cax.mirror.PVS.TP_01_MON,
        CAX_PREFIX + cax.mirror.PVS.TP_02_MON,
        CAX_PREFIX + cax.mirror.PVS.TP_03_MON,
        CAX_PREFIX + cax.mirror.PVS.TP_04_MON,
    ]

    # Pressure.
    PVPRESS = [
        CAX_PREFIX + cax.mirror.PVS.PR_A1_MON,
        CAX_PREFIX + cax.mirror.PVS.PR_A2_MON,
        CAX_PREFIX + cax.mirror.PVS.PR_A3_MON,
        CAX_PREFIX + cax.mirror.PVS.PR_A4_MON,

        CAX_PREFIX + cax.mirror.PVS.PR_Q1_MON,
        CAX_PREFIX + cax.mirror.PVS.PR_Q2_MON,
        CAX_PREFIX + cax.mirror.PVS.PR_Q3_MON,
        CAX_PREFIX + cax.mirror.PVS.PR_Q4_MON,

        CAX_PREFIX + cax.mirror.PVS.PR_V1_MON,
        CAX_PREFIX + cax.mirror.PVS.PR_V2_MON,
    ]

    # CAX environment PVs dictionary (temperature, water flux etc.).
    CAX_ENV_PVS = {
        "Mirror 1 input flow"        : PVFLUX[0],
        "Mirror 1 output flow"       : PVFLUX[1],
        "Cold Finger temp"           : PVTEMP[0],
        "Braid Mirror temp"          : PVTEMP[1],
        "Bar Braid Cold Finger temp" : PVTEMP[2],
        "Peltier Cold Side temp"     : PVTEMP[3],
        "Peltier Hot Side temp"      : PVTEMP[4],
        "Mirror 1 pressure"          : PVPRESS[0],
        "Mirror 2 pressure"          : PVPRESS[1],
        "Mirror 3 pressure"          : PVPRESS[2],
        "Mirror 4 pressure"          : PVPRESS[3],
        "Q1 pressure"                : PVPRESS[4],
        "Q2 pressure"                : PVPRESS[5],
        "Q3 pressure"                : PVPRESS[6],
        "Q4 pressure"                : PVPRESS[7],
        "V1 pressure"                : PVPRESS[8],
        "V2 pressure"                : PVPRESS[9],
        }
