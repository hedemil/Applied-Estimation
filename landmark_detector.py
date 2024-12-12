import numpy as np

class LandmarkDetector:
    def __init__(self):
        """
        Initialize the LandmarkDetector with a pre-computed angle array.
        The angles array corresponds to each laser reading angle in radians,
        spanning from -90 to +90 degrees in 0.5 degree increments.
        """
        self.angles = np.deg2rad(np.arange(-90, 90.5, 0.5))  # 361 angles in radians
        
    def find_16x(self, x):
        """
        Utility function equivalent to MATLAB's Find16X.
        Finds non-zero elements and converts indices to 16-bit integers.
        
        Args:
            x: Boolean numpy array
        Returns:
            Array of indices where x is True, as 16-bit integers
        """
        return np.where(x)[0].astype(np.int16)
    
    def detect_trees(self, RR):
        """
        Detect trees from laser range data using a multi-stage filtering process.
        
        Args:
            RR: numpy array of shape (361,) containing range measurements in meters
               Each measurement corresponds to an angle in self.angles
        
        Returns:
            x: numpy array of shape (3, n) where n is number of detected trees
               x[0, :] = distance to tree center (meters)
               x[1, :] = angle to tree center (radians)
               x[2, :] = tree diameter (meters)
        """

        RR = np.array(RR)
    
        # Ensure RR is 1D
        if len(RR.shape) > 1:
            RR = RR.flatten()

        # # Print debugging info
        # print("RR shape:", RR.shape)

        # Define detection parameters and thresholds
        M11 = 75      # Maximum valid range (meters)
        M10 = 1       # Minimum valid range (meters)
        daa = 5*np.pi/306  # Minimum angle difference for valid detection
        M2 = 1.5      # Maximum allowed range difference between consecutive points
        M2a = 10*np.pi/360  # Maximum allowed angle difference between consecutive points
        M3 = 3        # Maximum distance for clustering points
        M5 = 1        # Maximum allowed size for potential tree
        daMin2 = 2*np.pi/360  # Minimum angle difference for separate objects
        M3c = M3*M3   # Squared clustering threshold (avoid sqrt operations)
        
        RR = RR / 100
        # Stage 1: Initial Range Filtering
        # Filter out points beyond maximum range
        # Initial filtering
        ii1 = self.find_16x(RR < M11)
        L1 = len(ii1)
        
        # print("L1:", L1)  # Debug print
        
        if L1 < 1:
            return np.array([]).reshape(3, 0)

        R1 = RR[ii1]
        A1 = self.angles[ii1]

        # Find discontinuities
        ii2 = np.where((np.abs(np.diff(R1)) > M2) | (np.diff(A1) > M2a))[0]
        L2 = len(ii2) + 1

        # print("L2:", L2)  # Debug print
        # print("ii2 shape:", ii2.shape)  # Debug print

        # Make sure ii2u doesn't exceed array bounds
        ii2u = np.minimum(np.append(ii2, L1).astype(np.int16), L1-1)
        ii2 = np.concatenate(([0], ii2 + 1)).astype(np.int16)
        
        # print("ii2u shape:", ii2u.shape)  # Debug print
        # print("max ii2u:", np.max(ii2u))  # Debug print
        # print("R1 shape:", R1.shape)  # Debug print
        
        # Convert polar to Cartesian coordinates for easier geometric calculations
        R2 = R1[ii2]
        A2 = A1[ii2]
        R2u = R1[ii2u]
        A2u = A1[ii2u]
        
        x2 = R2 * np.cos(A2)    # x coordinates of lower points
        y2 = R2 * np.sin(A2)    # y coordinates of lower points
        x2u = R2u * np.cos(A2u) # x coordinates of upper points
        y2u = R2u * np.sin(A2u) # y coordinates of upper points
        
        # Stage 3: Clustering
        # Initialize clustering flags
        flag = np.zeros(L2, dtype=int)  # Track points to be merged/removed
        L3 = 0  # Counter for clustered points
        
        # Perform hierarchical clustering
        if L2 > 1:
            # Check adjacent points
            L2m = L2 - 1
            dx2 = x2[1:L2] - x2u[:L2m]  # x differences
            dy2 = y2[1:L2] - y2u[:L2m]  # y differences
            dl2 = dx2*dx2 + dy2*dy2      # Squared distances
            
            # Find close points
            ii3 = np.where(dl2 < M3c)[0]
            L3 = len(ii3)
            
            # Mark points for clustering
            if L3 > 0:
                flag[ii3] = 1
                flag[ii3 + 1] = 1
                
            # Check points with one point between them
            if L2 > 2:
                L2m = L2 - 2
                dx2 = x2[2:L2] - x2u[:L2m]
                dy2 = y2[2:L2] - y2u[:L2m]
                dl2 = dx2*dx2 + dy2*dy2
                
                ii3 = np.where(dl2 < M3c)[0]
                L3b = len(ii3)
                
                if L3b > 0:
                    flag[ii3] = 1
                    flag[ii3 + 2] = 1
                    L3 += L3b
                    
                # Check points with two points between them
                if L2 > 3:
                    L2m = L2 - 3
                    dx2 = x2[3:L2] - x2u[:L2m]
                    dy2 = y2[3:L2] - y2u[:L2m]
                    dl2 = dx2*dx2 + dy2*dy2
                    
                    ii3 = np.where(dl2 < M3c)[0]
                    L3b = len(ii3)
                    
                    if L3b > 0:
                        flag[ii3] = 1
                        flag[ii3 + 3] = 1
                        L3 += L3b
        
        # Stage 4: Angular Resolution Check
        # Handle objects that are close in angle but different in range
        if L2 > 1:
            ii3 = np.arange(L2-1)
            ii3 = ii3[A2[ii3 + 1] - A2u[ii3] < daMin2]  # Find close angles
            L3b = len(ii3)
            
            if L3b > 0:
                # Keep the closer object when objects overlap in angle
                ff = R2[ii3 + 1] > R2u[ii3]  # True if second point is further
                ii3 = ii3 + ff
                flag[ii3] = 1  # Mark the further point for removal
                L3 += L3b
        
        # Stage 5: Point Selection
        # Process remaining points after clustering
        if L3 > 0:
            # Select untagged points
            ii3 = self.find_16x(flag == 0)
            L3 = len(ii3)
            # Convert indices to floating point for later averaging
            ii4 = ii2[ii3].astype(float)
            ii4u = ii2u[ii3].astype(float)
            # Extract corresponding coordinates
            R4 = R2[ii3]
            R4u = R2u[ii3]
            A4 = A2[ii3]
            A4u = A2u[ii3]
            x4 = x2[ii3]
            y4 = y2[ii3]
            x4u = x2u[ii3]
            y4u = y2u[ii3]
        else:
            # Use all points if no clustering occurred
            ii4 = ii2.astype(float)
            ii4u = ii2u.astype(float)
            R4 = R2
            R4u = R2u
            A4 = A2
            A4u = A2u
            x4 = x2
            y4 = y2
            x4u = x2u
            y4u = y2u
        
        # Stage 6: Size Filtering
        # Filter based on object size
        dx2 = x4 - x4u
        dy2 = y4 - y4u
        dl2 = dx2*dx2 + dy2*dy2  # Squared size
        
        # Keep only reasonably sized objects
        ii5 = self.find_16x(dl2 < (M5*M5))
        L5 = len(ii5)
        if L5 < 1:
            return np.array([]).reshape(3, 0)
            
        # Extract filtered measurements
        R5 = R4[ii5]
        R5u = R4u[ii5]
        A5 = A4[ii5]
        A5u = A4u[ii5]
        ii4 = ii4[ii5]
        ii4u = ii4u[ii5]
        
        # Stage 7: Angle and Range Validation
        # Final filtering based on range and angle criteria
        ii5 = self.find_16x((R5 > M10) & (A5 > daa) & (A5u < (np.pi - daa)))
        L5 = len(ii5)
        if L5 < 1:
            return np.array([]).reshape(3, 0)
            
        # Extract validated measurements
        R5 = R5[ii5]
        R5u = R5u[ii5]
        A5 = A5[ii5]
        A5u = A5u[ii5]
        ii4 = ii4[ii5]
        ii4u = ii4u[ii5]
        
        # Stage 8: Tree Parameter Calculation
        # Calculate tree diameters and validate symmetry
        dL5 = (A5u + np.pi/360 - A5) * (R5 + R5u)/2  # Approximate diameter
        compa = np.abs(R5 - R5u) < (dL5/3)  # Symmetry check
        
        # Final filtering based on symmetry
        ii5 = self.find_16x(compa)
        L5 = len(ii5)
        if L5 < 1:
            return np.array([]).reshape(3, 0)
            
        # Extract final measurements
        R5 = R5[ii5]
        R5u = R5u[ii5]
        A5 = A5[ii5]
        A5u = A5u[ii5]
        ii4 = ii4[ii5]
        ii4u = ii4u[ii5]
        dL5 = dL5[ii5]
        
        # Stage 9: Final Position Calculation
        # Calculate final tree positions using interpolation
        auxi = (ii4 + ii4u)/2  # Average indices
        iia = np.floor(auxi).astype(int)  # Lower index
        iib = np.ceil(auxi).astype(int)   # Upper index
        Rs = (R1[iia] + R1[iib])/2        # Interpolated range
        
        # Return final tree parameters
        x = np.vstack((
            Rs + dL5/2,      # Final distance to tree center
            (A5 + A5u)*0.5,  # Final angle to tree center
            dL5              # Final tree diameter
        ))
        
        return x