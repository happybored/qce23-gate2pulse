import enum
import warnings
from math import pi, erf,sqrt,exp  # pylint: disable=no-name-in-module
from typing import List, Tuple, Union
import sys
import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
    Play,
    Schedule,
    ScheduleBlock,
    ControlChannel,
    DriveChannel,
    GaussianSquare,
    Waveform,
)
from qiskit.pulse import builder
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap


from qiskit.transpiler.passes.calibration.base_builder import CalibrationBuilder
from qiskit.transpiler.passes.calibration.exceptions import CalibrationNotAvailable

import scipy.integrate as integrate
from scipy.integrate import quad

def get_value(t,duration,width,sigma,amplitude):
    if t <= duration/2 - width/2:
       return amplitude*(-exp(-(-duration/2 + width/2 - 1)**2/(2*sigma**2)) + exp(-(-duration/2 + t + width/2)**2/(2*sigma**2)))/(1 - exp(-(-duration/2 + width/2 - 1)**2/(2*sigma**2)))
    elif t >= duration/2 + width/2:
       return amplitude*(-exp(-(duration/2 - width/2 + 1)**2/(2*sigma**2)) + exp(-(-duration/2 + t - width/2)**2/(2*sigma**2)))/(1 - exp(-(duration/2 - width/2 + 1)**2/(2*sigma**2)))
    else:
        return amplitude 

def get_area(duration,width,sigma,amplitude):
    return quad(get_value, 0, duration, args=(duration,width,sigma,amplitude))
    

# @staticmethod
def rescale_cr_fix_area2(inpulse, stretch_ratio: float, sample_mult: int = 16, vobose = False) -> int:
    """A builder macro to play stretched pulse.
    Args:
        instruction: The instruction from which to create a new shortened or lengthened pulse.
        theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
            play instruction implements.
        sample_mult: All pulses must be a multiple of sample_mult.
    Returns:
        Duration of stretched pulse.
    Raises:
        QiskitError: if rotation angle is not assigned.
    """
    # This method is called for instructions which are guaranteed to play GaussianSquare pulse
    params = inpulse.parameters.copy()
    risefall = (params["duration"] - params["width"])
    c1= exp(-pow((-1-risefall/2),2)/pow(params["sigma"],2)/2)
    norm_risefall_area = params["sigma"] * np.sqrt(2 * pi) * erf(risefall/params["sigma"]/(2*sqrt(2)))
    ori_norm_area =params["width"] + norm_risefall_area



    stretch_risefall = risefall*stretch_ratio
    c2= exp(-pow((-1-stretch_risefall/2),2)/pow(params["sigma"],2)/2)
    stretch_norm_risefall_area = params["sigma"] * np.sqrt(2 * pi) * erf(stretch_risefall/params["sigma"]/(2*sqrt(2)))
    stretch_width = params["width"] * stretch_ratio


    k1 =1/(1-c1)
    k2 =1/(1-c2)
    d1 =c1*k1* params["duration"]
    d2 =c2*k2* (stretch_width + stretch_risefall)
    stretch_norm_area = stretch_width + stretch_norm_risefall_area
    stretch_amplitude = params["amp"]*( ori_norm_area  *k1-d1)/ (stretch_norm_area*k2-d2)
    # print(params["amp"] )
    # print(stretch_amplitude)
    if abs(stretch_amplitude) >1:
        return  inpulse, params["duration"]
    else:

        area1 = get_area(params["duration"],params["width"],params["sigma"],params["amp"])
        params["amp"] = stretch_amplitude
        params["width"] = stretch_width
        print(stretch_width + stretch_risefall)
        round_duration = round((stretch_width + stretch_risefall) / sample_mult) * sample_mult
        params["duration"] = round_duration
        print(round_duration)
        area2 = get_area(params["duration"],params["width"],params["sigma"],params["amp"])
        if vobose:
            print("{}---{}".format(area1,area2))


        # sys.exit(0)
        # Only for test.
        # round_risefall = round_duration-stretch_width
        # round_risefall_sigma_ratio = round_risefall/ params["sigma"]
        # round_area = (params["sigma"] * np.sqrt(2 * pi) * erf(round_risefall_sigma_ratio) + params["width"])*params["amp"] 
        # print("round area 2 = ",round_area)

        stretched_pulse = GaussianSquare(**params)
        # print(stretched_pulse.area())
        # sys.exit(0)
        return stretched_pulse,round_duration
    


# @staticmethod
def rescale_cr_fix_area(inpulse, stretch_ratio: float, sample_mult: int = 16,vobose=False) -> int:
    """A builder macro to play stretched pulse.
    Args:
        instruction: The instruction from which to create a new shortened or lengthened pulse.
        theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
            play instruction implements.
        sample_mult: All pulses must be a multiple of sample_mult.
    Returns:
        Duration of stretched pulse.
    Raises:
        QiskitError: if rotation angle is not assigned.
    """
    # This method is called for instructions which are guaranteed to play GaussianSquare pulse
    params = inpulse.parameters.copy()
    risefall = (params["duration"] - params["width"])
    c1= exp(-pow((-1-risefall/2),2)/pow(params["sigma"],2)/2)
    norm_risefall_area = params["sigma"] * np.sqrt(2 * pi) * erf(risefall/params["sigma"]/(2*sqrt(2)))
    ori_norm_area =params["width"] + norm_risefall_area



    stretch_risefall = risefall* stretch_ratio
    stretch_width = params["width"] * stretch_ratio
    
    round_duration = round((stretch_width + stretch_risefall) / sample_mult) * sample_mult
    round_stretch_ratio = round_duration/params["duration"]
    stretch_sigma = params["sigma"]*round_stretch_ratio
    stretch_risefall = risefall* round_stretch_ratio
    stretch_width = params["width"] * round_stretch_ratio
    
    c2= exp(-pow((-1-stretch_risefall/2),2)/pow(stretch_sigma,2)/2)
    stretch_norm_risefall_area = stretch_sigma * np.sqrt(2 * pi) * erf(stretch_risefall/stretch_sigma/(2*sqrt(2)))
    


    k1 =1/(1-c1)
    k2 =1/(1-c2)
    d1 =c1*k1* params["duration"]
    d2 =c2*k2* (stretch_width + stretch_risefall)
    stretch_norm_area = stretch_width + stretch_norm_risefall_area
    stretch_amplitude = params["amp"]*( ori_norm_area  *k1-d1)/ (stretch_norm_area*k2-d2)
    if abs(stretch_amplitude) >1:
        return  inpulse, params["duration"]
    else:
        area1 = get_area(params["duration"],params["width"],params["sigma"],params["amp"])
        params["amp"] = stretch_amplitude
        params["width"] = stretch_width
        params["duration"] = round_duration
        params["sigma"] =stretch_sigma
        area2 = get_area(params["duration"],params["width"],params["sigma"],params["amp"])

        if vobose:
            print("{}---{}".format(area1,area2))
        # sys.exit(0)
        # Only for test.
        # round_risefall = round_duration-stretch_width
        # round_risefall_sigma_ratio = round_risefall/ params["sigma"]
        # round_area = (params["sigma"] * np.sqrt(2 * pi) * erf(round_risefall_sigma_ratio) + params["width"])*params["amp"] 
        # print("round area 2 = ",round_area)

        stretched_pulse = GaussianSquare(**params)
        # print(stretched_pulse.area())
        # sys.exit(0)
        return stretched_pulse,round_duration

class CXCalType(enum.Enum):
    """Estimated calibration type of backend CX gate."""

    ECR = "Echoed Cross Resonance"
    DIRECT_CX = "Direct CX"


class RZXStretchCalibrationBuilder(CalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate. This is done by retrieving (for a given pair of
    qubits) the CX schedule in the instruction schedule map of the backend defaults.
    The CX schedule must be an echoed cross-resonance gate optionally with rotary tones.
    The cross-resonance drive tones and rotary pulses must be Gaussian square pulses.
    The width of the Gaussian square pulse is adjusted so as to match the desired rotation angle.
    If the rotation angle is small such that the width disappears then the amplitude of the
    zero width Gaussian square pulse (i.e. a Gaussian) is reduced to reach the target rotation
    angle. Additional details can be found in https://arxiv.org/abs/2012.11660.
    """

    def __init__(
        self,
        stretch_ratio:float = 1,
        instruction_schedule_map: InstructionScheduleMap = None,
        qubit_channel_mapping: List[List[str]] = None,
        verbose: bool = True,
    ):
        """
        Initializes a RZXGate calibration builder.

        Args:
            instruction_schedule_map: The :obj:`InstructionScheduleMap` object representing the
                default pulse calibrations for the target backend
            qubit_channel_mapping: The list mapping qubit indices to the list of
                channel names that apply on that qubit.
            verbose: Set True to raise a user warning when RZX schedule cannot be built.

        Raises:
            QiskitError: Instruction schedule map is not provided.
        """
        super().__init__()

        if instruction_schedule_map is None:
            raise QiskitError("Calibrations can only be added to Pulse-enabled backends")

        if qubit_channel_mapping:
            warnings.warn(
                "'qubit_channel_mapping' is no longer used. This value is ignored.",
                DeprecationWarning,
            )

        self._inst_map = instruction_schedule_map
        self._verbose = verbose
        self.stretch_ratio = stretch_ratio

    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return isinstance(node_op, RZXGate) and self._inst_map.has("cx", qubits)

    @staticmethod
    @builder.macro
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int = 16) -> int:
        """A builder macro to play stretched pulse.

        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            Duration of stretched pulse.

        Raises:
            QiskitError: if rotation angle is not assigned.
        """
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        # This method is called for instructions which are guaranteed to play GaussianSquare pulse
        params = instruction.pulse.parameters.copy()
        risefall_sigma_ratio = (params["duration"] - params["width"]) / params["sigma"]

        # The error function is used because the Gaussian may have chopped tails.
        # Area is normalized by amplitude.
        # This makes widths robust to the rounding error.
        risefall_area = params["sigma"] * np.sqrt(2 * pi) * erf(risefall_sigma_ratio/(2*sqrt(2)))
        full_area = params["width"] + risefall_area

        # Get estimate of target area. Assume this is pi/2 controlled rotation.
        cal_angle = pi / 2
        target_area = abs(theta) / cal_angle * full_area
        new_width = target_area - risefall_area

        if new_width >= 0:
            width = new_width
            params["amp"] *= np.sign(theta)
        else:
            width = 0
            params["amp"] *= np.sign(theta) * target_area / risefall_area

        round_duration = (
            round((width + risefall_sigma_ratio * params["sigma"]) / sample_mult) * sample_mult
        )
        params["duration"] = round_duration
        params["width"] = width

        stretched_pulse = GaussianSquare(**params)
        builder.play(stretched_pulse, instruction.channel)

        return round_duration

    @staticmethod
    @builder.macro
    def rescale_cr_amplitude_inst(instruction: Play, theta: float, stretch_ratio: float, sample_mult: int = 16,vobose = False) -> int:
        """A builder macro to play stretched pulse.

        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            Duration of stretched pulse.

        Raises:
            QiskitError: if rotation angle is not assigned.
        """
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        # This method is called for instructions which are guaranteed to play GaussianSquare pulse
        params = instruction.pulse.parameters.copy()
        risefall_sigma_ratio = (params["duration"] - params["width"]) / params["sigma"]

        # The error function is used because the Gaussian may have chopped tails.
        # Area is normalized by amplitude.
        # This makes widths robust to the rounding error.
        risefall_area = params["sigma"] * np.sqrt(2 * pi) * erf(risefall_sigma_ratio/(2*sqrt(2)))
        full_area = params["width"] + risefall_area

        # Get estimate of target area. Assume this is pi/2 controlled rotation.
        cal_angle = pi / 2
        target_area = abs(theta) / cal_angle * full_area
        new_width = target_area - risefall_area

        if new_width >= 0:
            width = new_width
            params["amp"] *= np.sign(theta)
        else:
            width = 0
            params["amp"] *= np.sign(theta) * target_area / risefall_area

        round_duration = (
            round((width + risefall_sigma_ratio * params["sigma"]) / sample_mult) * sample_mult
        )
        params["duration"] = round_duration
        params["width"] = width

        # Only for test.
        # target_whole_area = (risefall_area + params["width"])*params["amp"] 
        # round_risefall = round_duration-width
        # round_risefall_sigma_ratio = round_risefall/ params["sigma"]
        # round_area = (params["sigma"] * np.sqrt(2 * pi) * erf(round_risefall_sigma_ratio) + params["width"])*params["amp"] 

        # print("target area = ",target_whole_area)
        # print("round area 1 = ",round_area)
        stretched_pulse = GaussianSquare(**params)

  
        stretched_pulse,round_duration = rescale_cr_fix_area(stretched_pulse,stretch_ratio, sample_mult,vobose)
        builder.play(stretched_pulse, instruction.channel)
        return round_duration
    
    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        """Builds the calibration schedule for the RZXGate(theta) with echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]

        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        if np.isclose(theta, 0.0):
            return ScheduleBlock(name="rzx(0.000)")

        cx_sched = self._inst_map.get("cx", qubits=qubits)
        cal_type, cr_tones, comp_tones = _check_calibration_type(cx_sched)

        if cal_type != CXCalType.ECR:
            if self._verbose:
                warnings.warn(
                    f"CX instruction for qubits {qubits} is likely {cal_type.value} sequence. "
                    "Pulse stretch for this calibration is not currently implemented. "
                    "RZX schedule is not generated for this qubit pair.",
                    UserWarning,
                )
            raise CalibrationNotAvailable

        if len(comp_tones) == 0:
            raise QiskitError(
                f"{repr(cx_sched)} has no target compensation tones. "
                "Native CR direction cannot be determined."
            )

        # Determine native direction, assuming only single drive channel per qubit.
        # This guarantees channel and qubit index equality.
        if comp_tones[0].channel.index == qubits[1]:
            xgate = self._inst_map.get("x", qubits[0])
            with builder.build(
                default_alignment="sequential", name="rzx(%.3f)" % theta
            ) as rzx_theta_native:
                for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                    with builder.align_left():
                        self.rescale_cr_amplitude_inst(cr_tone, theta,self.stretch_ratio)
                        self.rescale_cr_amplitude_inst(comp_tone, theta,self.stretch_ratio)
                    builder.call(xgate)
            return rzx_theta_native
        
        print('The direction is not native. Add Hadamard gates to flip the direction.')
        # The direction is not native. Add Hadamard gates to flip the direction.
        xgate = self._inst_map.get("x", qubits[1])
        szc = self._inst_map.get("rz", qubits[1], pi / 2)
        sxc = self._inst_map.get("sx", qubits[1])
        szt = self._inst_map.get("rz", qubits[0], pi / 2)
        sxt = self._inst_map.get("sx", qubits[0])
        with builder.build(name="hadamard") as hadamard:
            # Control qubit
            builder.call(szc, name="szc")
            builder.call(sxc, name="sxc")
            builder.call(szc, name="szc")
            # Target qubit
            builder.call(szt, name="szt")
            builder.call(sxt, name="sxt")
            builder.call(szt, name="szt")

        with builder.build(
            default_alignment="sequential", name="rzx(%.3f)" % theta
        ) as rzx_theta_flip:
            builder.call(hadamard, name="hadamard")
            for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                with builder.align_left():
                    self.rescale_cr_amplitude_inst(cr_tone, theta,self.stretch_ratio)
                    self.rescale_cr_amplitude_inst(comp_tone, theta,self.stretch_ratio)
                builder.call(xgate)
            builder.call(hadamard, name="hadamard")
        return rzx_theta_flip





class RZXStretchCalibrationBuilderNoEcho(RZXStretchCalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate.

    The ``RZXCalibrationBuilderNoEcho`` is a variation of the
    :class:`~qiskit.transpiler.passes.RZXCalibrationBuilder` pass
    that creates calibrations for the cross-resonance pulses without inserting
    the echo pulses in the pulse schedule. This enables exposing the echo in
    the cross-resonance sequence as gates so that the transpiler can simplify them.
    The ``RZXCalibrationBuilderNoEcho`` only supports the hardware-native direction
    of the CX gate.
    """

    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        """Builds the calibration schedule for the RZXGate(theta) without echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified,
                or the backend does not natively support the specified direction of the cx.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]
        # print()
        # print(theta)

        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        if np.isclose(theta, 0.0):
            return ScheduleBlock(name="rzx(0.000)")

        cx_sched = self._inst_map.get("cx", qubits=qubits)
        cal_type, cr_tones, comp_tones = _check_calibration_type(cx_sched)

        if cal_type != CXCalType.ECR:
            if self._verbose:
                warnings.warn(
                    f"CX instruction for qubits {qubits} is likely {cal_type.value} sequence. "
                    "Pulse stretch for this calibration is not currently implemented. "
                    "RZX schedule is not generated for this qubit pair.",
                    UserWarning,
                )
            raise CalibrationNotAvailable

        if len(comp_tones) == 0:
            raise QiskitError(
                f"{repr(cx_sched)} has no target compensation tones. "
                "Native CR direction cannot be determined."
            )

        # Determine native direction, assuming only single drive channel per qubit.
        # This guarantees channel and qubit index equality.
        if comp_tones[0].channel.index == qubits[1]:
            with builder.build(default_alignment="left", name="rzx(%.3f)" % theta) as rzx_theta:
                stretched_dur = self.rescale_cr_amplitude_inst(cr_tones[0], 2 * theta,self.stretch_ratio)
                self.rescale_cr_amplitude_inst(comp_tones[0], 2 * theta,self.stretch_ratio)
                # vobose = False
                # Placeholder to make pulse gate work
                builder.delay(stretched_dur, DriveChannel(qubits[0]))
            return rzx_theta

        raise QiskitError("RZXCalibrationBuilderNoEcho only supports hardware-native RZX gates.")


def _filter_cr_tone(time_inst_tup):
    """A helper function to filter pulses on control channels."""
    valid_types = ["GaussianSquare"]

    _, inst = time_inst_tup
    if isinstance(inst, Play) and isinstance(inst.channel, ControlChannel):
        pulse = inst.pulse
        if isinstance(pulse, Waveform) or pulse.pulse_type in valid_types:
            return True
    return False


def _filter_comp_tone(time_inst_tup):
    """A helper function to filter pulses on drive channels."""
    valid_types = ["GaussianSquare"]

    _, inst = time_inst_tup
    if isinstance(inst, Play) and isinstance(inst.channel, DriveChannel):
        pulse = inst.pulse
        if isinstance(pulse, Waveform) or pulse.pulse_type in valid_types:
            return True
    return False


def _check_calibration_type(cx_sched) -> Tuple[CXCalType, List[Play], List[Play]]:
    """A helper function to check type of CR calibration.

    Args:
        cx_sched: A target schedule to stretch.

    Returns:
        Filtered instructions and most-likely type of calibration.

    Raises:
        QiskitError: Unknown calibration type is detected.
    """
    cr_tones = list(
        map(lambda t: t[1], filter_instructions(cx_sched, [_filter_cr_tone]).instructions)
    )
    comp_tones = list(
        map(lambda t: t[1], filter_instructions(cx_sched, [_filter_comp_tone]).instructions)
    )

    if len(cr_tones) == 2 and len(comp_tones) in (0, 2):
        # ECR can be implemented without compensation tone at price of lower fidelity.
        # Remarkable noisy terms are usually eliminated by echo.
        return CXCalType.ECR, cr_tones, comp_tones

    if len(cr_tones) == 1 and len(comp_tones) == 1:
        # Direct CX must have compensation tone on target qubit.
        # Otherwise, it cannot eliminate IX interaction.
        return CXCalType.DIRECT_CX, cr_tones, comp_tones

    raise QiskitError(
        f"{repr(cx_sched)} is undefined pulse sequence. "
        "Check if this is a calibration for CX gate."
    )
