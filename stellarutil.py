import os, gizmo_analysis as gizmo, astropy.io.ascii as ascii, numpy as np, halo_analysis as halo

'''
NOTICE:
If a specific parameter/attribute is absent from "ROCKSTAR" but present under "AMIGA", it is not used in Rockstar.
This applies vice-versa.
'''

class Star:

    def __init__(self, x = 0, y = 0, z = 0, m = 0, a = 0, vx = 0, vy = 0, vz = 0):
        """
        Initialize a new Star object.

        Parameters:
        ----------
        x : float
            The x position of the star.
        y : float
            The y position of the star.
        z : float
            The z position of the star.
        m : float
            The mass of the star.
        a : float
            The scale factor of the star.
        vx : float
            The x velocity of the star.
        vy : float
            The y velocity of the star.
        vz : float
            The z velocity of the star.

        Attributes:
        -----------
        x : float
            The x position of the star.
        y : float
            The y position of the star.
        z : float
            The z position of the star.
        m : float
            The mass of the star.
        a : float
            The scale factor of the star.
        vx : float
            The x velocity of the star.
        vy : float
            The y velocity of the star.
        vz : float
            The z velocity of the star.
        velocity : float
            The velocity of the star.
        """
        self.x = x
        self.y = y
        self.z = z
        self.m = m
        self.a = a
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.velocity = self.get_velocity()
    
    def get_velocity(self):
        """
        Get the velocity of the star by calculating the magnitude of the velocity vector.
        
        Returns
        -------
        The velocity of the star.
        """
        return np.sqrt(np.square(self.vx) + np.square(self.vy)+ np.square(self.vz))
    
    def get_3DR(self):
        """
        Get the 3d radius of the star from the center of the halo.
        
        Returns
        -------
        The radius (r) of the star.
        """
        return np.sqrt(np.square(self.x) + np.square(self.y)+ np.square(self.z))

    def get_2DR(self):
        """
        Get the 2d radius of the star from the center of the halo.
        
        Returns
        -------
        The radius (r) of the star.
        """
        return np.sqrt(np.square(self.x) + np.square(self.y))
    
    def __str__(self):
        """
        The toString method for converting the object to a string.
        
        Returns
        -------
        A stringified version of the object.
        """
        output = f"Star:\n  Position: ({self.x}, {self.y}, {self.z}) [kpc]\n  Mass: {self.m} [unit]\n  Scale Factor (a): {self.a} [unit]\n  Velocity: {self.velocity()} [kpc/s]"
        return output

class Halo:

    def __init__(self, simulation, id, stars, dark_particles, xc, yc, zc, vxc, vyc, vzc, hostID, mass, radius, rMax, vMax, vEsc, numGas, gasMass, numStars, starMass, isRockstar):
        """
        Initialize a new Halo object.

        Parameters:
        ----------
        simulation : Simulation
            The simulation the halo comes from.
        id : integer
            The id of the halo.
        stars : Stars list
            The list of stars in the halo.
        dark_particles : Dark list
            The list of dark matter particles in the halo
        xc : float
            The center x position.
        yc : float
            The center y position.
        zc : float
            The center z position.
        hostID : integer
            The id of the parent halo.
        mass : float
            The mass of the halo.
        radius : float
            The radius of the halo.
        rMax : float
            The max radius of the halo.
        vMax : float
            The max velocity of the halo.
        vEsc : float
            The escape velocity of the halo.
        numGas : integer
            The number of gas particles.
        gasMass : float
            The mass of gas particles.
        numStars : integer
            The number of star particles.
        gasMass : float
            The mass of star particles.
        isRockstar: boolean
            Determines which logic to execute
        """
        self.simulation = simulation
        self.id = id
        self.stars = stars
        self.dark_particles = dark_particles
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.vxc = vxc
        self.vyc = vyc
        self.vzc = vzc
        self.hostID = hostID
        self.mass = mass
        self.radius = radius
        self.rMax = rMax
        self.vMax = vMax
        self.vEsc = vEsc
        self.numGas = numGas
        self.gasMass = gasMass
        self.numStars = numStars
        self.starMass = starMass
        self.isRockstar = isRockstar

    def restrict_percentage(self, percentage = 15):
        # AMIGA
        if self.isRockstar == False:
            # Get the radius of the galaxy that can actually hold stars
            # Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir
            rgal = (percentage / 100.0) * self.simulation.get_field('12')[self.id]
            # Get all the stars and center on the given halo
            x = self.simulation.particles['star']['position'][:,0] - self.xc
            y = self.simulation.particles['star']['position'][:,1] - self.yc
            z = self.simulation.particles['star']['position'][:,2] - self.zc
            a = self.simulation.particles['star']['form.scalefactor']
            m = self.simulation.particles['star']['mass']
            vx = self.simulation.particles['star']['velocity'][:,0] - self.vxc
            vy = self.simulation.particles['star']['velocity'][:,1] - self.vyc
            vz = self.simulation.particles['star']['velocity'][:,2] - self.vzc
            # Get the distance of each star from the center of the indicated dark matter halo
            distances =  np.sqrt(np.square(x) + np.square(y) + np.square(z))
            # Filter out all stars that are too far away 
            x_gal = x[distances < rgal]
            y_gal = y[distances < rgal]
            z_gal = z[distances < rgal]
            a_gal = a[distances < rgal]
            m_gal = m[distances < rgal]
            vx_gal = vx[distances < rgal]
            vy_gal = vy[distances < rgal]
            vz_gal = vz[distances < rgal]
            # Create a new stars list
            new_stars = []
            for i in range(len(x_gal)):
                star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
                new_stars.append(star)
            # Update the halos star list
            self.stars = new_stars

        # ROCKSTAR
        elif self.isRockstar == True:
            # Get the radius of the galaxy that can actually hold stars
            # Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir
            rgal = (percentage / 100.0) * self.simulation.hal['radius'][self.id]
            # Get all the stars and center on the given halo
            x = self.simulation.particles['star']['position'][:,0] - self.xc
            y = self.simulation.particles['star']['position'][:,1] - self.yc
            z = self.simulation.particles['star']['position'][:,2] - self.zc
            a = self.simulation.particles['star']['form.scalefactor']
            m = self.simulation.particles['star']['mass']
            vx = self.simulation.particles['star']['velocity'][:,0] - self.vxc
            vy = self.simulation.particles['star']['velocity'][:,1] - self.vyc
            vz = self.simulation.particles['star']['velocity'][:,2] - self.vzc
            # Get the distance of each star from the center of the indicated dark matter halo
            distances =  np.sqrt(np.square(x) + np.square(y) + np.square(z))
            # Filter out all stars that are too far away 
            x_gal = x[distances < rgal]
            y_gal = y[distances < rgal]
            z_gal = z[distances < rgal]
            a_gal = a[distances < rgal]
            m_gal = m[distances < rgal]
            vx_gal = vx[distances < rgal]
            vy_gal = vy[distances < rgal]
            vz_gal = vz[distances < rgal]
            # Create a new stars list
            new_stars = []
            for i in range(len(x_gal)):
                star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
                new_stars.append(star)
            # Update the halos star list
            self.stars = new_stars

            # If dark matter particles were specified in the simulation initialization, we perform the same operations for them...
            # Dark radius should be restricted by 100
            if self.dark_particles:
                dark_gal = (100.0 / 100.0) * self.simulation.hal['radius'][self.id]
                # Get all the dark matter particles and center on the given halo
                x_dark = self.simulation.particles['dark']['position'][:,0] - self.xc
                y_dark = self.simulation.particles['dark']['position'][:,1] - self.yc
                z_dark = self.simulation.particles['dark']['position'][:,2] - self.zc
                m_dark = self.simulation.particles['dark']['mass']
                # Get the distance of each dark matter particle from the center of the indicated dark matter halo
                distances_dark =  np.sqrt(np.square(x_dark) + np.square(y_dark) + np.square(z_dark))
                # Filter out all stars that are too far away 
                x_dark_gal = x_dark[distances_dark < dark_gal]
                y_dark_gal = y_dark[distances_dark < dark_gal]
                z_dark_gal = z_dark[distances_dark < dark_gal]
                m_dark_gal = m_dark[distances_dark < dark_gal]
                # Create a new dark matter particles list
                dark_particles_new = []
                for i in range(len(x_dark_gal)):
                    dark = Dark(x_dark_gal[i], y_dark_gal[i], z_dark_gal[i], m_dark_gal[i])
                    dark_particles_new.append(dark)
                # Update the dark particles list
                self.dark_particles = dark_particles_new


    def restrict_slice(self, face = 'xy', proj_distance = 1, thickness = 1):
        x = self.simulation.particles['star']['position'][:,0] - self.xc
        y = self.simulation.particles['star']['position'][:,1] - self.yc
        z = self.simulation.particles['star']['position'][:,2] - self.zc
        a = self.simulation.particles['star']['form.scalefactor']
        m = self.simulation.particles['star']['mass']
        vx = self.simulation.particles['star']['velocity'][:,0] - self.vxc
        vy = self.simulation.particles['star']['velocity'][:,1] - self.vyc
        vz = self.simulation.particles['star']['velocity'][:,2] - self.vzc
        # Get sqrt of the given face
        roots = []
        for i in range(len(x)):
            if face == 'xy' or face == 'yx': roots.append( (x**2 + y**2) ** .5 )
            elif face == 'xz' or face == 'zx': roots.append( (x**2 + z**2) ** .5 )
            elif face == 'yz' or face == 'zy': roots.append( (y**2 + z**2) ** .5 )
        # restrict all stars whose: sqrt(×^2 + y^2) < proj_distance
        x_gal = x[roots < proj_distance]
        y_gal = y[roots < proj_distance]
        z_gal = z[roots < proj_distance]
        a_gal = a[roots < proj_distance]
        m_gal = m[roots < proj_distance]
        vx_gal = vx[roots < proj_distance]
        vy_gal = vy[roots < proj_distance]
        vz_gal = vz[roots < proj_distance]
        # Create a new stars list
        new_stars = []
        for i in range(len(x_gal)):
            star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
            # restrict all stars whose |z pos| < thickness
            if face == 'xy' or face == 'yx':
                if abs(z_gal[i]) < thickness: new_stars.append(star)
            elif face == 'xz' or face == 'zx':
                if abs(y_gal[i]) < thickness: new_stars.append(star)
            elif face == 'yz' or face == 'zy':
                if abs(x_gal[i]) < thickness: new_stars.append(star)
        # Update the halos star list
        self.stars = new_stars

    def center_on(self, otherID):
        # AMIGA
        if self.isRockstar == False:
            # Get the center relative to the halo at the given index
            xc = (self.simulation.get_field('Xc(6)')[self.id]) - (self.simulation.get_field('Xc(6)')[otherID])
            yc = (self.simulation.get_field('Yc(7)')[self.id]) - (self.simulation.get_field('Yc(7)')[otherID])
            zc = (self.simulation.get_field('Zc(8)')[self.id]) - (self.simulation.get_field('Zc(8)')[otherID])
            # Recenter each star in the list
            for star in self.stars:
                star.x -= xc
                star.y -= yc
                star.z -= zc

        # ROCKSTAR
        if self.isRockstar == True:
            # Get the true index for the other halo
            other_index = self.simulation.sorted_keys[otherID]
            # Get the center relative to the halo at the given index
            xc = ( (self.simulation.hal['position'][self.id][0]) - (self.simulation.hal['position'][other_index][0]) )
            yc = ( (self.simulation.hal['position'][self.id][1]) - (self.simulation.hal['position'][other_index][1]) )
            zc = ( (self.simulation.hal['position'][self.id][2]) - (self.simulation.hal['position'][other_index][2]) )
            # Recenter each star in the list
            for star in self.stars:
                star.x -= xc
                star.y -= yc
                star.z -= zc
            # Recenter each dark matter particle in the list
            if self.dark_particles:
                for dark in self.dark_particles:
                    dark.x -= xc
                    dark.y -= yc
                    dark.z -= zc

class Dark:

    def __init__(self, x = 0, y = 0, z = 0, m = 0):
        """
        Initialize a new Dark object.

        Parameters:
        ----------
        x : float
            The x position of the dark matter particle.
        y : float
            The y position of the dark matter particle.
        z : float
            The z position of the dark matter particle.
        m : float
            The mass of the dark matter particle.

        Attributes:
        -----------
        x : float
            The x position of the dark matter particle.
        y : float
            The y position of the dark matter particle.
        z : float
            The z position of the dark matter particle.
        m : float
            The mass of the dark matter particle.
        """

        self.x = x
        self.y = y
        self.z = z
        self.m = m

class Simulation:

    def __init__(
            self, 
            simulation_name = None,
            simulation_directory = None, 
            snapshot_directory = 'output',
            ahf_path = None, 
            species = ['star'], 
            snapshot_value_kind='index',
            snapshot_values = 600,
            is_rockstar = False
        ):
        """
        Initialize a new Simulation object.

        Parameters:
        ----------
        # AMIGA #
        simulation_name : string
            The name of the simulation. 
            By giving the name, it will look for simulation_directory/snapshot_directory/ahf_directory in '../data/{simulation_name}'
        simulation_directory : string
            The path to the .hdf5 file. 
        snapshot_directory : string
            The path to the snapshot_times.txt. 
        ahf_path : string
            The path to the .AHF_halos file.
        species : list
            name[s] of particle species:
                'all' = all species in file
                'dark' = dark matter at highest resolution
                'dark2' = dark matter at lower resolution
                'gas' = gas
                'star' = stars
                'blackhole' = black holes, if snapshot contains them
        snapshot_value_kind : str
            What kind of value we are using for the snapshot. 
            For example, snapshot_value_kind of 'index' and snap_value of 600 means we are using the 600th snapshot from the simulation. 
            Another example would be snapshot_value_kind of 'redshift' and value of 0 means we are using a snapshot where redshift was 0.
        snapshot_values : int or float or list
            index[s] or redshift[s] or scale-factor[s] of snapshot[s]
        isRockstar : boolean
            User-specified boolean that indicates if a Rockstar sim is being read

        # ROCKSTAR #
        simulation_name : string
            Name of simulation to store for future identification, by giving name, it assumes correct file hiearchy
        simulation_directory : string
            Base directory of simulation / This should be the location of your snapshot_times.txt
        species : list
                Name[s] of particle species to read + assign to halos
                'star' : Must have star_NNN.hdf5
                'gas'  : Must have gas_NNN.hdf5
                'dark' : Must have dark_NNN.hdf5
                Note: Snapshot can be of any value (e.g. star_450.hdf5, gas_200.hdf5, and so forth)
        snapshot_value_kind : str
            What kind of value we are using for the snapshot. 
            For example, snapshot_value_kind of 'index' and snap_value of 600 means we are using the 600th snapshot from the simulation. 
            Another example would be snapshot_value_kind of 'redshift' and value of 0 means we are using a snapshot where redshift was 0.
        snapshot_values : int or float or list
            index[s] or redshift[s] or scale-factor[s] of snapshot[s]
        isRockstar : boolean
            User-specified boolean that indicates if a Rockstar sim is being read

        Attributes:
        -----------
        # AMIGA #
        h : float
            The hubble constant.
        h_fields : list
            List containing the fields that require division by h (Hubble constant)
        particles : float
            The data for all the indicated particles in the simulation.
        ahf_data : float
            The data within the .AHF_halos file.
        isRockstar : boolean
            Determines if Rockstar-specific logic should be used
        
        # ROCKSTAR #
        hal : dictionary class or list thereof
            Catalog[s] of halos at snapshot[s]
        h : float
            Hubble constant
        particles : list
            The data for all the indicated particles in the simulation
        sorted_keys : list
            List of halo indices sorted by mass in descending order (most to least massive)
        totalHalos : int
            Total number of halos (used so we can check if a get_halo() retrieval is valid)
        isRockstar : boolean
            Determines if Rockstar-specific logic should be used
        """

        # First, check if reading in a Rockstar or Amiga-processed simulation snapshot
        self.is_rockstar = is_rockstar

        # AMIGA 
        if self.is_rockstar == False:
            print("Amiga simulation selected...\n")
            # If a simulation name has been given, we can assume the user is using the conventional locations
            if simulation_name is not None:
                # Base directory of simulation
                simulation_directory = f'../data/{simulation_name}'
                # Tentatitive path to AHF_halos file (will resolve without issue when following correct file structure)
                ahf_test_path = os.path.join(simulation_directory, 'analysis/ahf/')
                # Look for the file that ends with '.AHF_halos'.
                items = os.listdir(ahf_test_path)
                for item in items:
                    file_path = os.path.join(ahf_test_path, item)
                    if not os.path.isdir(file_path) and item.endswith('.AHF_halos'):
                        print(file_path)
                        # Set the AHF_path now that we know an AHF_halos file exists
                        ahf_path = file_path
                        # When searching for the raw snapshot, program will join simulation directory path and this '/output'
                        snapshot_directory = 'output/'
                if ahf_path is None:
                    print(f'Could not find an ahf_directory in: {simulation_directory}')
                    return
            # Simulation directory path was provided (Ex. ../myDirectory/m10q_r250)
            elif simulation_directory is not None and ahf_path is None:
                # Tentative path to AHF_halos file (will resolve without issue when following correct file structure)
                ahf_test_path = os.path.join(simulation_directory, 'analysis/ahf/')
                # Look for the file that ends with '.AHF_halos'.
                items = os.listdir(ahf_test_path)
                for item in items:
                    file_path = os.path.join(ahf_test_path, item)
                    if not os.path.isdir(file_path) and item.endswith('.AHF_halos'):
                        # Set the AHF_path now that we know an AHF_halos file exists
                        ahf_path = file_path
                        print('Found AHF file here: ' + ahf_path)
                        # When searching for the raw snapshot, program will join simulation directory path and this '/output'
                        snapshot_directory = 'output/'
                        break
                if ahf_path is None:
                    print(f'Could not find an ahf_directory in: {simulation_directory}')
                    return
            else:
                if simulation_directory is None or ahf_path is None:
                    print('Cannot read files. Either:\n')
                    print('\t1) Provide a simulation_name while adhering to the proper folder structure.')
                    print('\t\tExample:  sim = Simulation("m10r_res250md")')
                    print('\t2) Manually specify: simulation_directory and ahf_directory. Also, specify the snapshot directory if it is not output.')
                    print('\t\tExample:  sim = Simulation(simulation_directory="path", ahf_path="path")')
                    print('\t\tExample:  sim = Simulation(simulation_directory="path", ahf_path="path", snapshot_directory="path")\n')
                    if simulation_directory is None:
                        print('Missing simulation directory.') 
                    if ahf_path is None:
                        print('Missing ahf_path.') 
                    return
          
            # Snpashot value is used to get the hubble constant, it will always be a subset of the snapshot_values
            snapshot_value = snapshot_values[0] if type(snapshot_values) is list else snapshot_values
            # Get the hubble constant from gizmo_analysis
            self.h = gizmo.io.Read.read_header(
                simulation_directory = simulation_directory,
                snapshot_directory = snapshot_directory,
                snapshot_value_kind = snapshot_value_kind,
                snapshot_value = snapshot_value
            )['hubble']
            # Initialize fields that require division by h (Hubble constant)
            self.h_fields = ['xc(6)', 'yc(7)', 'zc(8)', '12', '4', 'rmax(13)', 'mstar(65)', 'ngas(45)']
            # Get the particles from gizmo_analysis
            self.particles = gizmo.io.Read.read_snapshots(
                simulation_directory = simulation_directory,
                snapshot_directory = snapshot_directory,
                species=species, 
                snapshot_value_kind=snapshot_value_kind,
                snapshot_values=snapshot_values
            )
            # Get the AHF data from the halo file
            self.ahf_data = ascii.read(ahf_path)
            # Filter the AHF data
            self.ahf_data = self.ahf_data[(self.ahf_data.field('fMhires(38)') > 0.99)]

        # ROCKSTAR 
        elif self.is_rockstar == True:
            print("Rockstar simulation selected...\n")

            # Initialize variable to contain path to raw simulation snapshot hdf5 file (e.g. snapshot_600.hdf5)
            snapshot_path = None

            # Initialize variable to contain path to ROCKSTAR-produced halo hdf5 file (e.g. halo_600.hdf5)
            halo_path = None

            # Initialize variable to contain path to ROCKSTAR-produced species hdf5 file (e.g. star_600.hdf5)
            species_path = None

            # Initilaize variable to contain path to raw simulation snapshot directory (e.g. ../data/exampleSim/output/)
            output_dir_path = None

            # Initialize variable to contain the combined path that leads to ROCKSTAR hdf5 directory (simulation_directory + RCKSTAR_DIRECTORY + CTLG_HDF5_DIRECTORY)
            combined_dir_path = None

            # Set the subdirectory within the simulation_directory that will contain the raw simulation snapshot
            SNAPSHOT_DIRECTORY = 'output/'


            # Set the 3 directory paths required by halo_analysis (Rockstar, Catalog_HDF5, then Simulation).

            # Enter analysis folder within simulation directory where your "rockstar" subdirectory resides
            RCKSTAR_DIRECTORY= 'analysis/'
            # Enter "rockstar" subdirectory where your hdf5 files reside
            CTLG_HDF5_DIRECTORY = 'rockstar/'
            

            # If no directory has been given, we can assume the user is using the default directory configuration
            if simulation_directory is None:
                # Go up one directory level, then enter data folder, then search for folder matching simulation name
                simulation_directory = f'../data/{simulation_name}'
                # We will combine the path using os.join
                combined_dir_path = os.path.join(simulation_directory, RCKSTAR_DIRECTORY, CTLG_HDF5_DIRECTORY)
                # Also, we'll set path to raw simulation snapshot
                output_dir_path = os.path.join(simulation_directory, SNAPSHOT_DIRECTORY)       
            # Simulation directory has been provided
            elif simulation_directory is not None:
                # Enter rockstar folder within provided directory
                combined_dir_path = os.path.join(simulation_directory, RCKSTAR_DIRECTORY, CTLG_HDF5_DIRECTORY)
                # Also, we'll set path to raw simulation snapshot
                output_dir_path = os.path.join(simulation_directory, SNAPSHOT_DIRECTORY)

            # Look for the halo and species files (e.g. star_600) that end with '.hdf5'
            try:
                items = os.listdir(combined_dir_path)
            except FileNotFoundError:
                print(f'\nThe directory {combined_dir_path} does not exist.\n')
                print('Either:\n')
                print('\t1) Provide a simulation_name while adhering to the proper folder structure.')
                print('\t\tExample:  sim = Simulation("m10r_r250", is_rockstar = True)\n')
                print('\t2) Manually specify the simulation_directory')
                print('\t\tExample:  sim = Simulation(simulation_directory = "path", is_rockstar = True)\n')
                print('\tRecall that to work with halo_analysis, even the manually specified simulation directory must have its subdirectories '
                    + 'and files adhere to the halo_analysis structure.\n')
                return
            else:
                for item in items:
                    file_path = os.path.join(combined_dir_path, item)
                    # If the file path is not a directory and the file ends with .hdf5, we have found the files
                    if not os.path.isdir(file_path) and item.endswith('.hdf5'):
                        if "halo" in item:
                            print("\nFound ROCKSTAR halo .hdf5 file here: " + file_path)
                            halo_path = file_path
                        elif any(keyword in item for keyword in ['gas', 'star', 'dark']):
                            print("\nFound ROCKSTAR species .hdf5 file here: " + file_path)
                            species_path = file_path
                # If either (or both) type of .hdf5 file was not found, print an error message and return
                if halo_path is None and species_path is None:
                    print(f'\nCould not find any halo or species ROCKSTAR .hdf5 files in: {combined_dir_path}')
                    return
                elif halo_path is None:
                    print(f'\nCould not find any halo ROCKSTAR .hdf5 files in: {combined_dir_path}')
                    return
                elif species_path is None:
                    print(f'\nCould not find any species ROCKSTAR .hdf5 files in: {combined_dir_path}')
                    return
            
            # Look for the raw simulation snapshot file (e.g. snapshot_600.hdf5)
            try:
                items = os.listdir(output_dir_path)
            except FileNotFoundError:
                print(f'\nThe directory {output_dir_path} does not exist.\n')
                print('Within the simulation directory, there should be a folder named "output" that contains the raw simulation snapshot.\n')
                return
            else:
                for item in items:
                    file_path = os.path.join(output_dir_path, item)
                    # If the file path is not a directory and the file ends with .hdf5, we have found the file
                    if not os.path.isdir(file_path) and item.endswith('.hdf5'):
                        print("\nFound simulation snapshot .hdf5 file here: " + file_path)
                        snapshot_path = file_path
                if snapshot_path is None:
                    print(f'\nCould not find any simulation snapshot .hdf5 files in: {output_dir_path}')
                    return
                
            # Snpashot value is used to get the hubble constant, it will always be a subset of the snapshot_values
            snapshot_value = snapshot_values[0] if type(snapshot_values) is list else snapshot_values

            # Get the hubble constant from gizmo_analysis
            self.h = gizmo.io.Read.read_header(
                simulation_directory = simulation_directory,
                snapshot_directory = SNAPSHOT_DIRECTORY,
                snapshot_value_kind = snapshot_value_kind,
                snapshot_value = snapshot_value
            )['hubble']

            print(f'\nRetrieved hubble constant value of {self.h} from {snapshot_path}\n')

            # Read in raw snapshot(s) to access star/dark particle coordinates for scatterplots
            self.particles = gizmo.io.Read.read_snapshots(
                simulation_directory = simulation_directory,
                snapshot_directory=SNAPSHOT_DIRECTORY,
                species = species,
                snapshot_value_kind=snapshot_value_kind,
                snapshot_values=snapshot_values
            )
            

            print(f'Now printing self.particles (particle file types retrieved by gizmo_analysis)...')
            for key in self.particles.keys():
                print(key)

            # Read in catalog of halos from Rockstar snapshot
            self.hal = halo.io.IO.read_catalogs(snapshot_value_kind, snapshot_value, simulation_directory,
                                            rockstar_directory = RCKSTAR_DIRECTORY,
                                            catalog_hdf5_directory = CTLG_HDF5_DIRECTORY,
                                            species = species)
            
            # Reconstruct and perform fMHires
            print("\nNow performing fMHires...")

            # Print total halos before fMHires
            before_filter_total = len(self.hal['mass'])
            print(f"Total halos before fMHires: {before_filter_total}")

            # Indices of halos to keep
            filtered_indices = []

            # Calculate fraction of dark matter mass that is low resolution

            # Firsts, check if star particle file was provided either explicity or via "all"
            # If so, use dark2.mass as it is more accurate than mass.lowres
            if ('star' in species) or ('all' in species): 
                print("Star HDF5 retrieved... using dark2.mass for increased fMHires accuracy...")
                # For each halo in [total number of halos]
                for dm_halo in range(len(self.hal['mass'])):
                    low_res_frac = self.hal['dark2.mass'][dm_halo] / self.hal['mass'][dm_halo]
                    # Calculate the complement of low_res_frac (the value that when added, will equal 1)
                    hi_res_frac = 1 - low_res_frac
                    # Filter out halos whose hi_res_frac value exceeds 0.99 
                    if hi_res_frac > 0.99:
                        filtered_indices.append(dm_halo)
            else:
                print("Star HDF5 not provided... using mass.lowres thus reduced fMHires accuracy...")
                # For each halo in [total number of halos]
                for dm_halo in range(len(self.hal['mass'])):
                    low_res_frac = self.hal['mass.lowres'][dm_halo] / self.hal['mass'][dm_halo]
                    # Calculate the complement of low_res_frac (the value that when added, will equal 1)
                    hi_res_frac = 1 - low_res_frac
                    # Filter out halos whose hi_res_frac value exceeds 0.99 
                    if hi_res_frac > 0.99:
                        filtered_indices.append(dm_halo)

            # Remove halos whose hi_res_frac is less than or equal to 0.99 from the hal dictionary
            # First, retrieve the data from each key
            for key in self.hal.keys():
                data = self.hal[key]
                # Check if the data is a list or NumPy-like array
                if isinstance(data, list):
                    # Filter 1D lists
                    self.hal[key] = [data[i] for i in filtered_indices]
                elif hasattr(data, "shape"):
                    # Handle NumPy-like arrays
                    if len(data.shape) >= 1:
                        # Slice along the first axis, irrespective of its dimensionality
                        self.hal[key] = data[filtered_indices, ...]
                    else:
                        raise ValueError(f"Unsupported array shape for key '{key}': {data.shape}")
                else:
                    raise TypeError(f"Unsupported data type for key '{key}': {type(data)}")

            # Print total halos after filtering
            after_filter_total = len(self.hal["mass"])
            print(f"Total halos after fMHires: {after_filter_total}")

            # Print total halos removed
            print(f"Total halos filtered out: {before_filter_total - after_filter_total}")
            
            # Create a list of indices sorted by mass in descending order (most massive to least massive)
            self.sorted_keys = sorted(range(len(self.hal['mass'])), key=lambda x: self.hal['mass'][x], reverse=True)

            # Get the total number of halos so we can check if a get_halo() search is valid
            self.totalHalos = len(self.hal['mass'])

        else:
            # Print an error message
            print("A valid option for is_rockstar was not provided.")
            print("Please enter either True or False.")

        
    
    def get_halo(self, id = 0):
        """
        Get the indicated dark matter halo.

        Parameters:
        ----------
        id : int
            The index of the dark matter halo. Default is 0.
        
        Returns
        -------
        The indicated dark matter halo in a Halo object.
        """

        # AMIGA
        if self.is_rockstar == False:
            # Get the center of the indicated dark matter halo
            xc = self.ahf_data.field('Xc(6)')[id] / self.h
            yc = self.ahf_data.field('Yc(7)')[id] / self.h
            zc = self.ahf_data.field('Zc(8)')[id] / self.h
            # Get the peculiar velocity of the indicated dark matter halo
            vxc = self.ahf_data.field('VXc(9)')[id] / self.h
            vyc = self.ahf_data.field('VYc(10)')[id] / self.h
            vzc = self.ahf_data.field('VZc(11)')[id] / self.h
            # Get the x,y,z positions of each star particle in the simulation
            # And normalize it with the center of the indicated dark matter halo
            x = self.particles['star']['position'][:,0] - xc
            y = self.particles['star']['position'][:,1] - yc
            z = self.particles['star']['position'][:,2] - zc
            # Get the scalefactor (age) of each star in the simulation
            a = self.particles['star']['form.scalefactor']
            # Get the mass of each star in the simulation
            m = self.particles['star']['mass']
            # Get the x,y,z velocity of each star particle in the simulation
            # And normalize it with the peculiar velocity of the indicated dark matter halo
            vx = self.particles['star']['velocity'][:,0] - vxc
            vy = self.particles['star']['velocity'][:,1] - vyc
            vz = self.particles['star']['velocity'][:,2] - vzc
            # Get the distance of each star from the center of the indicated dark matter halo
            distances = np.sqrt(np.square(x) + np.square(y) + np.square(z))
            # Get the radius of the galaxy that can actually hold stars
            # Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir
            rgal = self.get_field('12')[id]
            # Filter out all stars that are too far away 
            x_gal = x[distances < rgal]
            y_gal = y[distances < rgal]
            z_gal = z[distances < rgal]
            a_gal = a[distances < rgal]
            m_gal = m[distances < rgal]
            vx_gal = vx[distances < rgal]
            vy_gal = vy[distances < rgal]
            vz_gal = vz[distances < rgal]
            # All the lists are the same length
            # Loop through and make a list of stars
            stars = []
            for i in range(len(x_gal)):
                star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
                stars.append(star)
            # Grab some more metadata for the halo
            hostID = self.ahf_data.field('hostHalo(2)')[id]
            mass = self.get_field('4')[id]
            rMax = self.ahf_data.field('Rmax(13)')[id] / self.h
            vMax = self.ahf_data.field('Vmax(17)')[id]
            vEsc = self.ahf_data.field('v_esc(18)')[id]
            numGas = self.ahf_data.field('n_gas(44)')[id]
            gasMass = self.ahf_data.field('M_gas(45)')[id]
            numStars = self.ahf_data.field('n_star(64)')[id]
            starMass = self.ahf_data.field('M_star(65)')[id]
            # Pass on the boolean that determines which logic to perform
            isRockstar = self.is_rockstar
            # Pass in a placeholder value for dark_particles when using Amiga
            dark_particles = 0
            # Return the indicated dark matter halo
            halo = Halo(self, id, stars, dark_particles, xc, yc, zc, vxc, vyc, vzc, hostID, mass, rgal, rMax, vMax, vEsc, numGas, gasMass, numStars, starMass, isRockstar)
            return halo
        
        # ROCKSTAR
        elif self.is_rockstar == True:
            # Check if id is within the valid range of halos
            if id < 0 or id >= self.totalHalos:
                raise IndexError(f"Index id is out of range. Please access a halo from 0 to {self.totalHalos - 1}.")
            # Get the true index for the desired halo
            halo_index = self.sorted_keys[id]
            # Get the center of the indicated dark matter halo
            # We do not divide by h as it is already done for us when the values are retrieved from halo_analysis
            xc = self.hal['position'][halo_index][0] # Xc
            yc = self.hal['position'][halo_index][1] # Yc
            zc = self.hal['position'][halo_index][2] # Zc
            # Get the peculiar velocity of the indicated dark matter halo
            # We do not divide by h as it is already done for us when the values are retrieved from halo_analysis
            vxc = self.hal['velocity'][halo_index][0] # VXc
            vyc = self.hal['velocity'][halo_index][1] # VYc
            vzc = self.hal['velocity'][halo_index][2] # VZc
            # Get the x,y,z positions of each star particle in the simulation
            #   Note: gizmo_analysis divides it by h for us
            # And normalize it with the center of the indicated dark matter halo
            x = self.particles['star']['position'][:,0] - xc
            y = self.particles['star']['position'][:,1] - yc
            z = self.particles['star']['position'][:,2] - zc
            # Get the scalefactor (age) of each star in the simulation
            a = self.particles['star']['form.scalefactor']
            # Get the mass of each star in the simulation
            m = self.particles['star']['mass']
            # Get the x,y,z velocity of each star particle in the simulation
            # And normalize it with the peculiar velocity of the indicated dark matter halo
            vx = self.particles['star']['velocity'][:,0] - vxc
            vy = self.particles['star']['velocity'][:,1] - vyc
            vz = self.particles['star']['velocity'][:,2] - vzc
            # Get the distance of each star from the center of the indicated dark matter halo
            distances = np.sqrt(np.square(x) + np.square(y) + np.square(z))
            # Get the radius of the galaxy that can actually hold stars
            # Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir
            rgal = self.hal['radius'][halo_index]
            # Filter out all stars that are too far away 
            x_gal = x[distances < rgal]
            y_gal = y[distances < rgal]
            z_gal = z[distances < rgal]
            a_gal = a[distances < rgal]
            m_gal = m[distances < rgal]
            vx_gal = vx[distances < rgal]
            vy_gal = vy[distances < rgal]
            vz_gal = vz[distances < rgal]
            # All the lists are the same length
            # Loop through and make a list of stars
            stars = []
            for i in range(len(x_gal)):
                star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
                stars.append(star)

            # If "dark" was specified during simulation initialization, then we gather dark particle information:
            dark_particles = []
            if 'dark' in self.particles.keys():
                print("Dark was specified when initializing simulation, therefore also retrieving dark particles during get_rockstar_halo()...\n")
                # Get the x,y,z positions of each star particle in the simulation
                #   Note: gizmo_analysis divides it by h for us
                # And normalize it with the center of the indicated dark matter halo
                x_dark = self.particles['dark']['position'][:,0] - xc
                y_dark = self.particles['dark']['position'][:,1] - yc
                z_dark = self.particles['dark']['position'][:,2] - zc
                # Get the mass of each dark matter particle in the simulation
                m_dark = self.particles['dark']['mass']
                # Get the distance of each dark matter particle from the center of the indicated dark matter halo
                distances_dark = np.sqrt(np.square(x_dark) + np.square(y_dark) + np.square(z_dark))
                # Filter out all stars that are too far away 
                x_dark_gal = x_dark[distances_dark < rgal]
                y_dark_gal = y_dark[distances_dark < rgal]
                z_dark_gal = z_dark[distances_dark < rgal]
                m_dark_gal = m_dark[distances_dark < rgal]
                # Loop through and make a list of dark matter particles
                for i in range(len(x_dark_gal)):
                    dark = Dark(x_dark_gal[i], y_dark_gal[i], z_dark_gal[i], m_dark_gal[i])
                    dark_particles.append(dark)

            # Grab some more metadata for the halo
            hostID = self.hal['host.index'][halo_index]
            mass = self.hal['mass'][halo_index]
            rMax = self.hal['radius'][halo_index] # We assign Rvir (virial radius) instead
            vMax = self.hal['vel.circ.max'][halo_index]

            # Give dummy values for the three below. 
            vEsc = None # Give it all -1's / or do None
            numGas = None # We need gas species file
            gasMass = None # We need gas species file

            numStars = self.hal['star.number']
            starMass = self.hal['star.mass'][halo_index]

            # Pass on the boolean that determines which logic to perform
            isRockstar = self.is_rockstar

            # Return the indicated dark matter halo
            halo = Halo(self, halo_index, stars, dark_particles, xc, yc, zc, vxc, vyc, vzc, hostID, mass, rgal, rMax, vMax, vEsc, numGas, gasMass, numStars, starMass, isRockstar)
            return halo
        

    def get_field(self, field, divide_h = True):
        """
        Get the values in the column of the specified field from the .AHF_halos file.

        Parameters:
        ----------
        # AMIGA #
        field : string
            The name of the field.

        divide_h : Boolean
            If you would like to divide by h (Hubble Constant).
            Automatically set to True.

        # ROCKSTAR #
        field : string
            The name of the field
        
        Returns
        -------
        The list of values in that field.
        """
        
        # AMIGA
        if self.is_rockstar == False:
            # Get the correct name of the field
            field_name = str(field).lower()  # Convert field to string if it's an integer
            field_name = field_name.replace('_','')
            # Loop through all the field names
            for item in self.ahf_data.dtype.names:
                string = item.lower().replace('_','')
                if field_name in string:
                    field_name = item
                    break 

            if divide_h == True:
                h_query = str(field).lower() # Convert field to string if its an integer
                for h_field in self.h_fields:
                    string = h_field.lower().replace('_','')
                    if h_query in string:
                        # Desired field is one where we must divide by h before returning
                        # Perform division by h
                        column = self.ahf_data.field(field_name) / self.h
                        return column
                    
                # True was passed but the field is one where we do not divide by h
                column = self.ahf_data.field(field_name)
                return column
            else: 
                # Store all the field data in a list called column
                column = self.ahf_data.field(field_name) 
                # Return the column
                return column
            
        # ROCKSTAR
        elif self.is_rockstar == True:
            # Get the correct name of the field
            field_name = str(field).lower()  # Convert field to string if it's an integer
            field_name = field_name.replace('_','')

            # Loop through all the field names
            for item in self.hal:
                string = str(item.lower().replace('_',''))
                if field_name == string:
                    return(self.hal[string])

    
    def help(self):
        '''
        Recieve help.
        '''
        def print_menu():
            # AMIGA
            if self.is_rockstar == False:
                print("---------------------------------------------------------------------------")
                print("a) What is a halo file (.AHF_halos)?")
                print("b) Print a list of fields from the halo file.")
                print("c) Clear this screen.")
                print("d) What is a simulation file (.hdf5)?")
                print("e) Print a list of fields from the particle file.")
                print("l) Print the list of libraries installed via pip3.")
                print("m) Print menu.")
                print("p) Print python paths.")
                print("q) Quit.")
                print("---------------------------------------------------------------------------")
            
            # ROCKSTAR
            if self.is_rockstar == True:
                print("---------------------------------------------------------------------------")
                print("a) What is ROCKSTAR?")
                print("b) Print all accessible halo properties.")
                print("c) Clear this screen.")
                print("l) Print the list of libraries installed via pip3.")
                print("m) Print menu.")
                print("p) Print python paths.")
                print("q) Quit.")
                print("---------------------------------------------------------------------------")

        def print_halo_fields():
            # AMIGA
            if self.is_rockstar == False:
                print("\tID(1) - halo ID") 
                print("\thostHalo(2) - ID of host halo, 0 (or -1) if halo itself is not a subhalo") 
                print("\tMvir(4) - mass of halo [Mo/h]") 
                print("\tXc(6) - x position of halo [kpc/h]") 
                print("\tYc(7) - y position of halo [kpc/h]") 
                print("\tZc(8) - z position of halo [kpc/h]") 
                print("\tVXc(9) - x peculiar velocity of halo [km/sec]") 
                print("\tVYc(10) - y peculiar velocity of halo [km/sec]") 
                print("\tVZc(11) - z peculiar velocity of halo [km/sec]") 
                print("\tRvir(12) - virial radius [kpc/h]") 
                print("\tRmax(13) - position of rotation curve maximum [kpc/h]") 
                print("\tVmax(17) - maximum of rotation curve [km/sec]") 
                print("\tv_esc(18) - escape velocity at Rvir [km/sec]") 
                print("\tn_gas(44) - number of gas particles in halo")
                print("\tM_gas(45) - mass of gas particles in halo")
                print("\tn_star(64) - number of star particles in halo")
                print("\tM_star(65) - mass of star particles in halo")
            
            # ROCKSTAR
            if self.is_rockstar == True:
                print("\tPrinting all halo properties accessible through simulation.get_field([desired_property])")
                for k in self.hal.keys():
                    print(k)

        print_menu()
        while True:
            prompt = input("\nPress 'm' to see options menu. Enter an option: ").lower()
            print()
            if prompt == 'q' or prompt == "quit":
                break
            else:
                if prompt == 'a':
                    # AMIGA
                    if self.is_rockstar == False:
                        print("\tThe halo file is a summary of the data from the simulation file (.hdf5).")
                        print("\tTo see the names of all relevant fields, enter 'b'")
                    # ROCKSTAR
                    elif self.is_rockstar == True:
                        print("\tRockstar is a halo finding algorithm just like Amiga Halo Finder (AHF)!")
                        print("\tCosmological simulations produce .hdf5 files which Rockstar is then run upon.")
                        print("\tRockstar will analyze the distribution of particles in the simulation to identify " +
                            "dark matter halos!")
                        print("\tRockstar will then produce its own .hdf5 files of both star particles (and sometimes gas and/or dark) and the " +
                            "identified dark matter halos.")
                elif prompt == 'b':
                    print_halo_fields()
                elif prompt == 'c':
                    os.system('clear')
                elif prompt == 'd':
                    # AMIGA
                    if self.is_rockstar == False:
                        print("\tThe simulation file contains information about all particles in the simulation.")
                        print("\tNeed to edit this stuff underneath when structure changes.")
                        print("\tCATEGORIES: star, gas")
                        print("\tTo see the names of all relevant fields, enter 'e'")
                        print("\tTo use: 'particles['CATEGORY']['FIELD']'")
                    # ROCKSTAR
                    elif self.is_rockstar == True:
                        print("\tYou have not chosen a valid option.")
                elif prompt == 'e':
                    # AMIGA
                    if self.is_rockstar == False:
                        print("\tStar - position, mass, massfraction, id.child, id.generation, id, form.scalefactor, velocity")  
                        print("\tGas - position, density, electron.fraction, temperature, mass, massfraction, hydrogen.neutral.fraction, id.child, id.generation, id, size, sfr, velocity")
                    # ROCKSTAR
                    elif self.is_rockstar == True:
                        print("You have not chosen a valid option.")  
                elif prompt == 'l':
                    os.system('pip3 list')
                elif prompt == 'm':
                    print_menu()
                elif prompt == 'p':
                    os.system("echo $PYTHONPATH | tr ':' '\n'")
                else:
                    print("\tYou have not chosen a valid option.")
                    