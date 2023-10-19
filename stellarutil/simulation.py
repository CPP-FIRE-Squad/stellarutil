import os, gizmo_analysis as gizmo, astropy.io.ascii as ascii, numpy as np
from stellarutil.calculations import dist, filter_list, dist2D
from stellarutil.console import help

#region talk to gizmo stuff

def get_ahf_data(path, filter = True):
    '''
    Read the content from the AHF file.
    
    Parameters:
    ----------
        path : str 
            the path to the file
        filter : boolean
            should the data be filtered to only include rows with: fMhires(38) > 0.99. Default is true.

    Returns
    -------
        Return a 2D list of the data specified in the AHF file.
    '''

    data = ascii.read(path)
    if not filter:
        return data
    else:
        data_filtered = data[(data.field('fMhires(38)') > 0.99)]
        return data_filtered

def get_field(data, field):
    '''
    Return a list of all items in the field of a dataset

    Parameters:
    ----------
        data : 2d array
            The table of data 
        field : int | string 
            The name of the field.
        elem_range : range
            Range of indices to print (default is all).

    Returns
    -------
        The ist of all items in the field.
    '''
    
    # Get the correct name of the field
    field_name = get_field_name(data, field)
    print(f"{field} changed to {field_name}")
    # Store all the field data in a list called column
    column = data.field(field_name) 
    # Return the column
    return column

def get_field_name(data, name):
    '''
    Return the correct name of a field.

    Parameters:
    ----------
        data : 2d array
            The table of data 
        name : int | string 
            The name of the field.

    Returns
    -------
        The ist of all items in the field.
    '''
    name = str(name).lower()  # Convert field to string if it's an integer
    name = name.replace('_','')
    # Loop through all the field names
    for item in data.dtype.names:
        string = item.lower().replace('_','')
        if name in string:
            return item
            
    return None

def get_hubble_constant(simulation_directory, snapshot_directory, snapshot_value, snapshot_value_kind):
    header = gizmo.io.Read.read_header(
        simulation_directory = simulation_directory,
        snapshot_directory = snapshot_directory,
        snapshot_value_kind = snapshot_value_kind,
        snapshot_value = snapshot_value
    )
    return header['hubble']

def get_particles(simulation_directory, snapshot_directory, species, snapshot_values, snapshot_value_kind):
    return gizmo.io.Read.read_snapshots(
        simulation_directory = simulation_directory,
        snapshot_directory = snapshot_directory,
        species=species, 
        snapshot_value_kind=snapshot_value_kind,
        snapshot_values=snapshot_values
    )

#endregion



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
        return dist(self.vx, self.vy, self.vz)
    
    def get_3DR(self):
        """
        Get the 3d radius of the star from the center of the halo.
        
        Returns
        -------
        The radius (r) of the star.
        """
        return dist(self.x, self.y, self.z)
    
    def get_2DR(self):
        """
        Get the 2d radius of the star from the center of the halo.
        
        Returns
        -------
        The radius (r) of the star.
        """
        return dist(self.x, self.y)
    
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

    def __init__(self, index, stars, xc, yc, zc, vxc, vyc, vzc):
        self.index = index
        self.stars = stars
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.vxc = vxc
        self.vyc = vyc
        self.vzc = vzc


class Simulation:

    def __init__(
            self, 
            simulation_name = None,
            simulation_directory = None, 
            snapshot_directory = 'output',
            ahf_path = None, 
            species = ['star'], 
            snapshot_value_kind='index',
            snapshot_values = 600
        ):
        """
        Initialize a new Simulation object.

        Parameters:
        ----------
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
        snapshot_values : int or float or list
            index[s] or redshift[s] or scale-factor[s] of snapshot[s]

        Attributes:
        -----------
        h : float
            The hubble constant.
        particles : float
            The data for all the indicated particles in the simulation.
        ahf_data : float
            The data within the .AHF_halos file.
        """
        # If a simulation name has been given, we can assume the user is using the conventional locations
        if simulation_name is not None:
            simulation_directory = f'../data/{simulation_name}'
            # Look for the file that ends with '.AHF_halos'.
            items = os.listdir(simulation_directory)
            for item in items:
                file_path = os.path.join(simulation_directory, item)
                if not os.path.isdir(file_path) and item.endswith('.AHF_halos'):
                    print(file_path)
                    ahf_path = file_path
            if ahf_path is None:
                print(f'Could not find an ahf_directory in: {simulation_directory}')
                return
        elif simulation_directory is not None and ahf_path is None:
            # Look for the file that ends with '.AHF_halos'.
            items = os.listdir(simulation_directory)
            for item in items:
                file_path = os.path.join(simulation_directory, item)
                if not os.path.isdir(file_path) and item.endswith('.AHF_halos'):
                    ahf_path = file_path
                    print('Found AHF file here: ' + ahf_path)
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
        # Get the data from gizmo_analysis
        self.h = get_hubble_constant(simulation_directory, snapshot_directory, snapshot_value, snapshot_value_kind)
        self.particles = get_particles(simulation_directory, snapshot_directory, species, snapshot_values, snapshot_value_kind)
        self.ahf_data = get_ahf_data(ahf_path)

    def get_halo(self, index = 0):
        """
        Get the list of stars inside an indicated dark matter halo.

        Parameters:
        ----------
        index : int
            The index of the dark matter halo. Default is 0.
        restrict : float
            The restriction percentage. Default is 0.15 (15%).
        
        Returns
        -------
        The list of all stars in the indicated dark matter halo.
        """
        # Get the center of the indicated dark matter halo
        xc = self.ahf_data.field('Xc(6)')[index] / self.h
        yc = self.ahf_data.field('Yc(7)')[index] / self.h
        zc = self.ahf_data.field('Zc(8)')[index] / self.h
        # Get the peculiar velocity of the indicated dark matter halo
        vxc = self.ahf_data.field('VXc(9)')[index] / self.h
        vyc = self.ahf_data.field('VYc(10)')[index] / self.h
        vzc = self.ahf_data.field('VZc(11)')[index] / self.h
        # Get the x,y,z positions of each star particle in the simulation
        # And normalize it with the center of the indicated dark matter halo
        x = self.particles['star']['position'][:,0] - xc
        y = self.particles['star']['position'][:,1] - yc
        z = self.particles['star']['position'][:,2] - zc

        # TODO - replace this pseudocode
        # x = x - np.mean(x)
        # y = y - np.mean(y)
        # z = z - np.mean(z)

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
        distances =  dist(x,y,z) 
        # Get the radius of the galaxy that can actually hold stars
        # Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir
        rgal = self.get_field('12')[index] / self.h 
        # Filter out all stars that are too far away 
        x_gal = filter_list(x, distances, rgal)
        y_gal = filter_list(y, distances, rgal)
        z_gal = filter_list(z, distances, rgal)
        a_gal = filter_list(a, distances, rgal)
        m_gal = filter_list(m, distances, rgal)
        vx_gal = filter_list(vx, distances, rgal)
        vy_gal = filter_list(vy, distances, rgal)
        vz_gal = filter_list(vz, distances, rgal)

        # All the lists are the same length
        # Loop through and make a list of stars
        stars = []
        for i in range(len(x_gal)):
            star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
            stars.append(star)

        # Return the indicated dark matter halo
        halo = Halo(index, stars, xc, yc, zc, vxc, vyc, vzc)
        return halo

    def restrict_percentage(self, halo, percentage = 15):
        # Get the radius of the galaxy that can actually hold stars
        # Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir
        rgal = (percentage / 100.0) * self.get_field('12')[halo.index] / self.h 
        # Get all the stars and center on the given halo
        x = self.particles['star']['position'][:,0] - halo.xc
        y = self.particles['star']['position'][:,1] - halo.yc
        z = self.particles['star']['position'][:,2] - halo.zc
        a = self.particles['star']['form.scalefactor']
        m = self.particles['star']['mass']
        vx = self.particles['star']['velocity'][:,0] - halo.vxc
        vy = self.particles['star']['velocity'][:,1] - halo.vyc
        vz = self.particles['star']['velocity'][:,2] - halo.vzc
        # Get the distance of each star from the center of the indicated dark matter halo
        distances =  dist(x,y,z) 
        # Filter out all stars that are too far away 
        x_gal = filter_list(x, distances, rgal)
        y_gal = filter_list(y, distances, rgal)
        z_gal = filter_list(z, distances, rgal)
        a_gal = filter_list(a, distances, rgal)
        m_gal = filter_list(m, distances, rgal)
        vx_gal = filter_list(vx, distances, rgal)
        vy_gal = filter_list(vy, distances, rgal)
        vz_gal = filter_list(vz, distances, rgal)
        # Create a new stars list
        new_stars = []
        for i in range(len(x_gal)):
            star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
            new_stars.append(star)
        # Update the halos star list
        halo.stars = new_stars

    def restrict_slice(self, halo, stars, face = 'xy', proj_distance = 1, thickness = 1):
        x = self.particles['star']['position'][:,0] - halo.xc
        y = self.particles['star']['position'][:,1] - halo.yc
        z = self.particles['star']['position'][:,2] - halo.zc
        a = self.particles['star']['form.scalefactor']
        m = self.particles['star']['mass']
        vx = self.particles['star']['velocity'][:,0] - halo.vxc
        vy = self.particles['star']['velocity'][:,1] - halo.vyc
        vz = self.particles['star']['velocity'][:,2] - halo.vzc
        # Get sqrt of the given face
        roots = []
        for i in range(len(x)):
            if face == 'xy' or face == 'yx': roots.append( (x**2 + y**2) ** .5 )
            elif face == 'xz' or face == 'zx': roots.append( (x**2 + z**2) ** .5 )
            elif face == 'yz' or face == 'zy': roots.append( (y**2 + z**2) ** .5 )

        # restrict all stars whose: sqrt(×^2 + y^2) < proj_distance
        x_gal = filter_list(x, roots, proj_distance)
        y_gal = filter_list(y, roots, proj_distance)
        z_gal = filter_list(z, roots, proj_distance)
        a_gal = filter_list(a, roots, proj_distance)
        m_gal = filter_list(m, roots, proj_distance)
        vx_gal = filter_list(vx, roots, proj_distance)
        vy_gal = filter_list(vy, roots, proj_distance)
        vz_gal = filter_list(vz, roots, proj_distance)
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
        halo.stars = new_stars


    def center_on(self, halo, center_index):
        # Get the center relative to the halo at the given index
        xc = self.get_field('Xc(6)')[halo.index] / self.h - self.get_field('Xc(6)')[center_index] / self.h
        yc = self.get_field('Yc(7)')[halo.index] / self.h - self.get_field('Yc(7)')[center_index] / self.h
        zc = self.get_field('Zc(8)')[halo.index] / self.h - self.get_field('Zc(8)')[center_index] / self.h
        # Recenter each star in the list
        for star in halo.stars:
            star.x -= xc
            star.y -= yc
            star.z -= zc

    def get_field(self, field):
        """
        Get the values in the column of the specified field from the .AHF_halos file.

        Parameters:
        ----------
        field : string
            The name of the field.
        
        Returns
        -------
        The list of values in that field.
        """
        # Get the correct name of the field
        field_name = get_field_name(self.ahf_data, field)
        # Store all the field data in a list called column
        column = self.ahf_data.field(field_name) 
        # Return the column
        return column
    
    def help(self):
        help()