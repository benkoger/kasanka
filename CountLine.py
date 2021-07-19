import numpy as np

class CountLine():
    ''' 
    Counts everytime a bat crosses a line defined in space (y = constant)
    '''
    
    def __init__(self, line_value, line_dim=1, total_frames=None):
        '''
        line_value: the value that defines the position of the 
            line the bats are crossing. 
        line_dim: whether the line is horizontal or verical 
            (0 for vertical, 1 for horizontal)
        total_frames: total_frames in video
        
        '''
        self.line_value = line_value
        self.line_dim = line_dim
        self.total_frames = total_frames
        if total_frames:
            # How many bats are crossing the line in each frame
            self.num_crossing = np.zeros(total_frames)
            self.forward = np.zeros(total_frames)
            self.backward = np.zeros(total_frames)
        self.bat_ids_crossed = []
        # What frames did crosses occur
        self.frame_cross = []
        
    def is_crossing(self, track, track_ind):
        '''
        Checks if given track is crossing the line.  
        
        track: a bat track
        track_ind: number track in list of tracks
        
        returns 1 is forward crossing
                -1 if backward crossing
                0 if no crossing
        '''
        
        # Crossing line going away   
        if track['track'][0, self.line_dim] >= self.line_value:
            # last frame was below line
            if track['track'][-1, self.line_dim] <= self.line_value:
                frame_num = (np.argmin(track['track'][-1, self.line_dim] <= self.line_value)
                             + track['first_frame'])
                # this frame above or on line
                # So bat has crossed line
                if self.total_frames:
                    self.num_crossing[frame_num] += 1
                    self.forward[frame_num] += 1
                    self.bat_ids_crossed.append(track_ind)
                    self.frame_cross.append(frame_num)

                return (1, frame_num)
        

        # Crossing line coming back      
        if track['track'][0, self.line_dim] <= self.line_value:
            # last frame was above line
            if track['track'][-1, self.line_dim] >= self.line_value:
                # this frame below or on line
                # So bat has crossed line coming back
                frame_num = (np.argmin(track['track'][-1, self.line_dim] >= self.line_value)
                             + track['first_frame'])
                if self.total_frames:
                    self.num_crossing[frame_num] -= 1
                    self.backward[frame_num] += 1
                    self.bat_ids_crossed.append(-track_ind)
                    self.frame_cross.append(frame_num)
                return (-1, frame_num)
        return (0, None) 