import bisect

class PitchConverter:
    freq_level = [
        65.41, 65.41, 73.42, 82.41, 87.31, 98.00, 110.00, 123.47, 130.81, 146.83, 164.81, 174.61,
        196.00, 220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
        587.33, 659.25, 698.46, 783.99, 880.00, 987.77, 1046.50, 1174.66, 1174.66
    ]

    pitch_string = [
        "C2", "C2", "D2", "E2", "F2", "G2", "A2", "B2", "C3", "D3", "E3", "F3", "G3", "A3", "B3",
        "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5", "A5", "B5",
        "C6", "D6", "D6"
    ]

    @staticmethod
    def freq_to_pitch(value):
        nearest_index = PitchConverter.search_nearest(value)
        if 0 <= nearest_index < len(PitchConverter.pitch_string):
            return PitchConverter.pitch_string[nearest_index]
        return PitchConverter.pitch_string[0]

    @staticmethod
    def search_nearest(value):
        pos = bisect.bisect_left(PitchConverter.freq_level, value)
        if pos == 0:
            return 0
        if pos == len(PitchConverter.freq_level):
            return len(PitchConverter.freq_level) - 1

        before = PitchConverter.freq_level[pos - 1]
        after = PitchConverter.freq_level[pos]

        if after - value < value - before:
            return pos
        return pos - 1