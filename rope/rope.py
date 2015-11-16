#!/bin/env python
"""Calculate a transition path with the modified string method.

Usage
-----
On deneb:
srun <script name>

Requirements
------------
- mpi4py version 1.3.1 or more recent
 (implies version 0.2.2. or more recent of ctypes)
- custom version of lammps.py

Synopsis
--------
This is a prototype implementation of the modified string method, see Weinan et
al. The Journal of Chemical Physics 126, 164103 2007.

The general  workflow is as  follows: First, MPI  communicators are set  up and
a  number  of  num_replicas  Lammps  instances  are  created  and  initialized.
Note  that more  than one  CPU  can be  associated with  each Lammps  instance,
i.e.  it  is  possible  to  parallelize  on  the  replica  level.  Second,  the
transition path is calculated iteratively.  The maximum number of iterations is
max_string_iterations. As  the first step  in each iteration, the  equations of
motion are integrated with Lammps (using num_lammps_iterations sub-iterations).
After  integration,  the atomic  positions  are  extracted  and the  string  is
reparameterized using linear interpolation.

Note  that  in the  time  splitting  scheme proposed  in  the  paper of  Weinan
and  coworkers,  the string  would  be  reparameterized after  every  timestep.
However, this  would be  computationally too expensive.  Note that  waiting for
num_lammps_timesteps between  reparameterizations will  reduce the  accuracy of
the calculation, because replicas will slide down the transition path, parallel
to the path.

Input and Controls
------------------
The  main  parameters are  presently  hard-coded,  right  at the  beginning  of
the  script.  The  Lammps  instructions  are passed  via  the  template  string
setup_template. Modify this  template according to your needs.  You can perform
subsitutions using the placeholder '?'. Currently, the starting coordinates for
the replicas are read from a set of files with the name pattern <i>.atom, where
i is the replica number (zero offset). These must be Lammps dump files, but you
can change that easily by modifying the template.

Author
------
Wolfram Georg Noehring
Tue Jul 14 22:46:59 CEST 2015
Fri Nov  6 15:22:24 CET 2015
"""

from mpi4py import MPI
import numpy as np
from numpy import ma as ma
from lammps import lammps
from string import Template
import configparser
import sys
import ctypes

def main():
    # Parse configuration file
    configfile = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(configfile)
    setup_file = config.get('setup', 'setup_file')
    num_replicas = config.getint('setup', 'num_replicas')
    if (num_replicas < 3):
        raise ValueError('need at least three replicas')
    num_atoms = config.getint('setup', 'num_atoms')
    num_dof = num_atoms * 3 # Degrees of freedom
    fixed_types = config.get('setup', 'fixed_types')
    fixed_types = [int(i) for i in fixed_types.split(' ')]
    periodic_boundary = config.get('setup', 'periodic_boundary')
    periodic_boundary = [int(i) for i in periodic_boundary.split(' ')]
    if len(periodic_boundary) != 3:
        raise ValueError('Wrong periodic boundary information')
    num_lammps_iterations = config.getint('run', 'num_lammps_iterations')
    max_string_iterations = config.getint('run', 'max_string_iterations')
    string_displacement_threshold = config.getfloat('run', 'string_displacement_threshold')
    measure_time = config.getboolean('run', 'measure_time')

    # Split communicator into groups. Each group will simulate one replica.
    world = MPI.COMM_WORLD
    if world.size < num_replicas - 1:
        raise ValueError(
                'Number of processes must be larger than number of replicas!'
        )
    replica_association = np.array_split(np.arange(world.size), num_replicas)
    for i, ranks in enumerate(replica_association):
        if world.rank in ranks:
            replica = i
            break
    del(replica_association)
    group = world.Split(color=replica, key=world.rank)
    is_inner_replica = int(0 < replica < num_replicas - 1)
    if group.rank == 0:
        logfile = open('string.log.{:d}'.format(replica), 'w')
    has_timer = measure_time and (world.rank == 0)

    # Get the ranks of the group-roots
    if group.rank == 0:
        flag = np.int(world.rank)
    else:
        flag = np.int(-1.0)
    group_root_flags = np.zeros(world.size, dtype=np.int)
    world.Allgather(
            sendbuf=[np.array(flag, dtype=np.int), MPI.INT],
            recvbuf=[group_root_flags, MPI.INT]
    )
    group_roots = np.empty(num_replicas, dtype=np.int)
    if world.rank == 0:
        group_roots = np.asarray([i for i in group_root_flags if i>-1], dtype=np.int)
    world.Bcast(buf=[group_roots, MPI.INT], root=0)
    if group.rank == 0:
        logfile.write('MPI group roots:\t' + list_to_str(group_roots) + '\n')

    # Determine left and right neighbors in a ring-like MPI topology
    left_replica = np.roll(range(num_replicas), +1)[replica]
    right_replica = np.roll(range(num_replicas), -1)[replica]
    left_root = group_roots[left_replica]
    right_root = group_roots[right_replica]
    if group.rank == 0:
        logfile.write(
                'Left and right MPI group root:\t{:d}\t{:d}\n'.format(
                    left_root, right_root
                )
        )
    # Declare arrays for neighbor data
    left_energy = np.empty(1, dtype=float)
    right_energy = np.empty(1, dtype=float)
    left_dof = np.empty(num_dof)
    right_dof = np.empty(num_dof)

    # Create Lammps instances
    if MPI._sizeof(group) == ctypes.sizeof(ctypes.c_int):
        MPI_Comm = ctypes.c_int
    else:
        MPI_Comm = ctypes.c_void_p
    comm_ptr = MPI._addressof(group)
    comm_val = MPI_Comm.from_address(comm_ptr)
    my_lammps = lammps(name="", cmdargs="", ptr=comm_val)
    if group.rank == 0:
        logfile.write('Generated Lammps instance\n')

    # Initialize replicas
    # Note: scatter_atoms requires an atom map
    with open(setup_file) as file:
        setup_template = file.read()
    setup_template = safe_template_string(setup_template)
    setup = setup_template.substitute(c = replica)
    for line in setup.splitlines():
        my_lammps.command(line)
    if group.rank == 0:
        logfile.write('Initialized replicas\n')

    # Divide the 3N-dimensional array of atomic degrees of freedom (dof)
    # and the array of image flags into chunks. Each chunk is associated
    # with a  particular MPI  rank in  the group.  The chunks  should be
    # approximately evenly-sized and  aligned such that the  dof of each
    # atom are all in the same chunk.
    ideal_dof_chunk_size = np.floor(num_dof / group.size).astype(int)
    ideal_dof_chunk_size -= ideal_dof_chunk_size%3
    dof_chunk_bounds = [
        (i * ideal_dof_chunk_size, (i + 1) * ideal_dof_chunk_size)
        for i in range(group.size - 1)
    ]
    dof_chunk_bounds += [((group.size - 1) * ideal_dof_chunk_size, num_dof)]
    # Displacements for MPI scattering
    dof_chunk_displ = [c[0] for c in dof_chunk_bounds]
    dof_chunk_sizes = [len(range(c[0], c[1])) for c in dof_chunk_bounds]
    # Chunk for image flags:
    ideal_img_chunk_size = ideal_dof_chunk_size / 3
    img_chunk_bounds = [
        (i * ideal_img_chunk_size, (i + 1) * ideal_img_chunk_size)
        for i in range(group.size - 1)
    ]
    img_chunk_bounds += [((group.size - 1) * ideal_img_chunk_size, num_atoms)]
    img_chunk_displ = [c[0] for c in img_chunk_bounds]
    img_chunk_sizes = [len(range(c[0], c[1])) for c in img_chunk_bounds]
    img_chunk = np.empty(img_chunk_sizes[group.rank], dtype=np.int32)
    if group.rank == 0:
        logfile.write('Chunks for degrees of freedom:\n')
        for i, bound in enumerate(dof_chunk_bounds):
            logfile.write(
                    '{:d}\t{:d}\t({:d} DOFs)\n'.format(
                    bound[0], bound[1], dof_chunk_sizes[i])
            )

    # Calculate size of the box
    box_size = np.zeros(3)
    box_size[0] += my_lammps.extract_global("boxxhi", 1)
    box_size[0] -= my_lammps.extract_global("boxxlo", 1)
    box_size[1] += my_lammps.extract_global("boxyhi", 1)
    box_size[1] -= my_lammps.extract_global("boxylo", 1)
    box_size[2] += my_lammps.extract_global("boxzhi", 1)
    box_size[2] -= my_lammps.extract_global("boxzlo", 1)
    if group.rank == 0:
        logfile.write('Box edge length:\t' + list_to_str(box_size) + "\n")

    # Declare chunks as masked
    # Note: masks are preserved in Scatterv / Gatherv operations.
    # They are not set by these operations.
    my_atom_types = np.asarray(my_lammps.gather_atoms("type", 0, 1))
    my_atom_types = np.vstack([my_atom_types] * 3)
    my_atom_types = my_atom_types.reshape((-1,),order='F')
    dof_mask = [my_atom_types == i for i in fixed_types]
    dof_mask = np.logical_or.reduce(dof_mask)
    fixed_dofs = np.where(dof_mask)[0]
    chunk_of_dof_mask = dof_mask[
            dof_chunk_bounds[group.rank][0]:dof_chunk_bounds[group.rank][1]
    ]
    chunk_of_my_dof = ma.empty(dof_chunk_sizes[group.rank])
    chunk_of_my_dof.mask = chunk_of_dof_mask
    chunk_of_my_dof_old = ma.empty(dof_chunk_sizes[group.rank])
    chunk_of_my_dof_old.mask = chunk_of_dof_mask
    if replica > 0:
        chunk_of_left_dof = ma.empty(dof_chunk_sizes[group.rank])
        chunk_of_left_dof.mask = chunk_of_dof_mask
        chunk_of_right_dof = ma.empty(dof_chunk_sizes[group.rank])
        chunk_of_right_dof.mask = chunk_of_dof_mask
    chunk_of_tangent = ma.empty(dof_chunk_sizes[group.rank])
    chunk_of_tangent.mask = chunk_of_dof_mask
    chunk_of_my_force = ma.empty(dof_chunk_sizes[group.rank])
    chunk_of_my_force.mask = chunk_of_dof_mask
    chunk_of_perpendicular_force = ma.empty(dof_chunk_sizes[group.rank])
    chunk_of_perpendicular_force.mask = chunk_of_dof_mask
    if group.rank == 0:
        logfile.write('Masked chunks\n')

    # Initialize old coordinates and unwrap
    my_dof_old = np.asarray(my_lammps.gather_atoms("x", 1, 3))
    my_image_flags = np.asarray(my_lammps.gather_atoms("image", 0, 1))
    group.Scatterv(
        [my_dof_old, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
        chunk_of_my_dof, root=0
    )
    group.Scatterv(
        [my_image_flags, img_chunk_sizes, img_chunk_displ, MPI.INT],
        img_chunk, root=0
    )
    image_distances = calc_image_distances(
        img_chunk, periodic_boundary, box_size
    )
    chunk_of_my_dof = apply_pbc(
        chunk_of_my_dof, periodic_boundary, image_distances, 'unwrap'
    )
    group.Gatherv(
        [chunk_of_my_dof, dof_chunk_sizes[group.rank], MPI.DOUBLE],
        [my_dof_old, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
        root=0
    )
    if group.rank == 0:
        logfile.write('Initialized old coordinates\n')

    converged = False
    world.Barrier()
    minimize_command = "minimize 0 1e-8 {:d} {:d}".format(
            num_lammps_iterations, num_lammps_iterations
    )
    if has_timer:
        timer = Timer()
    for iteration in range(max_string_iterations):
        if has_timer:
            timer.stamp()
        # Evolve replica
        my_lammps.command(minimize_command)
        my_lammps.command("velocity all set 0.0 0.0 0.0")
        # Extract coordinates and energy
        # Note: - all processes in the group have the same view
        #       - np.asarray from ctypes does not copy, ideally
        #       - these arrays are not masked
        my_dof = np.asarray(my_lammps.gather_atoms("x", 1, 3))
        my_image_flags = np.asarray(my_lammps.gather_atoms("image", 0, 1))
        my_force = np.asarray(my_lammps.gather_atoms("f", 1, 3))
        my_energy = np.asarray(my_lammps.extract_variable("PE", "all", 0))
        # Unwrap atom positions
        group.Scatterv(
            [my_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            chunk_of_my_dof, root=0
        )
        group.Scatterv(
            [my_image_flags, img_chunk_sizes, img_chunk_displ, MPI.INT],
            img_chunk, root=0
        )
        image_distances = calc_image_distances(
            img_chunk, periodic_boundary, box_size
        )
        chunk_of_my_dof = apply_pbc(
            chunk_of_my_dof, periodic_boundary, image_distances, 'unwrap'
        )
        group.Gatherv(
            [chunk_of_my_dof, dof_chunk_sizes[group.rank], MPI.DOUBLE],
            [my_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            root=0
        )
        energies = np.empty(world.size)
        world.Allgather(
                sendbuf=[my_energy, MPI.DOUBLE],
                recvbuf=[energies, MPI.DOUBLE]
        ) # Implicit world.Barrier
        energies = energies[group_roots]
        left_energy = np.roll(energies, +1)[replica]
        my_energy = energies[replica]
        right_energy = np.roll(energies, -1)[replica]
        if group.rank == 0:
            logfile.write('Minimized energy and applied pbc\n')
        if has_timer:
            timer.stamp()
            logfile.write(
                'Time since last timer call:\t' + str(timer.elapsed_time) + '\n'
            )

        # Get coordinates and energy of left and right neighbor
        if (group.rank == 0):
            world.Sendrecv(
                sendbuf=[my_dof, MPI.DOUBLE], dest=right_root,
                recvbuf=[left_dof, MPI.DOUBLE], source=left_root
            )
            world.Sendrecv(
                sendbuf=[my_dof, MPI.DOUBLE], dest=left_root,
                recvbuf=[right_dof, MPI.DOUBLE], source=right_root
            )

        # Compute distance to left neighbor
        if (replica > 0):
            group.Scatterv(
                sendbuf=[left_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                recvbuf=chunk_of_left_dof,
                root=0
            )
            group.Scatterv(
                sendbuf=[my_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                recvbuf=chunk_of_my_dof,
                root=0
            )
            chunk_dist_left, distance_to_left = calc_chunk_distance(
                    chunk_of_my_dof, chunk_of_left_dof, group)
        else:
            distance_to_left = 0.0

        # Todo: compute the force components also for the initial and the final replica, 
        # taking forward and backward differences
        # Compute the tangent vector and the force perpendicular to the path
        if is_inner_replica:
            if group.rank == 0:
                logfile.write(
                    'Tangent computation. Energies:\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                        left_energy, my_energy, right_energy
                    )
                )
            if (left_energy > my_energy > right_energy):
                if group.rank == 0:
                    logfile.write('Backward difference\n')
                chunk_of_tangent = chunk_dist_left / distance_to_left
            elif (left_energy < my_energy < right_energy):
                if group.rank == 0:
                    logfile.write('Forward difference\n')
                group.Scatterv(
                    [right_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                    chunk_of_right_dof, root=0)
                chunk_dist, norm_of_distance = calc_chunk_distance(
                        chunk_of_right_dof, chunk_of_my_dof, group)
                chunk_of_tangent = chunk_dist / norm_of_distance
            else:
                if group.rank == 0:
                    logfile.write('Central difference\n')
                group.Scatterv(
                    [right_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                    chunk_of_right_dof, root=0)
                chunk_dist, norm_of_distance = calc_chunk_distance(
                        chunk_of_right_dof, chunk_of_left_dof, group)
                chunk_of_tangent = chunk_dist / norm_of_distance
            group.Scatterv(
                [my_force, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                chunk_of_my_force, root=0)
            chunk_projection = ma.dot(
                    ma.ravel(chunk_of_my_force),
                    ma.ravel(chunk_of_tangent)
            )
            my_tangential_force = np.empty(1)
            group.Allreduce(
                [chunk_projection, MPI.DOUBLE],
                [my_tangential_force, MPI.DOUBLE], op=MPI.SUM)
            chunk_of_perpendicular_force, norm_of_perpendicular_force = calc_chunk_distance(
                    chunk_of_my_force, my_tangential_force * chunk_of_tangent, group)
            if group.rank == 0:
                logfile.write(
                    'Projection of force on tangent:\t{:.4e}\n'.format(
                        my_tangential_force[0]
                    )
                )
                logfile.write(
                    'Norm of perpendicular force:\t{:.4e}\n'.format(
                        norm_of_perpendicular_force[0]
                    )
                )

        # Todo: eliminate Allgather by sending from group roots to world rank 0 instead
        else:
            my_tangential_force = np.zeros(1)
            norm_of_perpendicular_force = np.zeros(1)
        tangential_forces = np.empty(world.size)
        world.Allgather(
                sendbuf=[my_tangential_force, MPI.DOUBLE],
                recvbuf=[tangential_forces, MPI.DOUBLE]
        )
        tangential_forces = tangential_forces[group_roots]
        perpendicular_forces = np.empty(world.size)
        world.Allgather(
                sendbuf=[norm_of_perpendicular_force, MPI.DOUBLE],
                recvbuf=[perpendicular_forces, MPI.DOUBLE]
        )
        perpendicular_forces = perpendicular_forces[group_roots]

        # Compute total length of string
        # Here, we would weight by energy. The factor 1.0 is a placeholder.
        if (replica > 0 and group.rank == 0):
            length_increment = 1.0 * distance_to_left
        distance_to_left = np.asarray(distance_to_left)
        world.Barrier()

        # Determine the positions of the replicas on the string
        parameterization = np.empty(world.size)
        world.Allgather(
                sendbuf=[distance_to_left, MPI.DOUBLE],
                recvbuf=[parameterization, MPI.DOUBLE]
        )
        parameterization = parameterization[group_roots]
        parameterization = np.cumsum(parameterization, out=parameterization)
        length = parameterization[-1]
        parameterization /= length
        # Calculate ideal replica positions.
        # If the current positions are energy-weighted, then then target
        # positions must be weighted, too.
        my_target_parameter = np.array(float(replica + 1) / float(num_replicas))
        target_parameterization = np.empty(world.size)
        world.Allgather(
                sendbuf=[np.array(my_target_parameter), MPI.DOUBLE],
                recvbuf=[target_parameterization, MPI.DOUBLE]
        )
        target_parameterization = target_parameterization[group_roots]
        if group.rank == 0:
            logfile.write(
                'Parameterization:\t' + list_to_str(parameterization) + '\n'
            )
            logfile.write(
                'Target parameterization:\t'
                + list_to_str(target_parameterization) + '\n'
            )
        if has_timer:
            timer.stamp()
            logfile.write(
                'Time since last timer call:\t' + str(timer.elapsed_time) + '\n'
            )

        # Determine which replica is associated with the interval
        # in which the new position lies and request interpolation
        bins = np.searchsorted(parameterization, target_parameterization, 'left')
        my_dest = np.where(bins == replica)[0]
        my_dest = [i for i in my_dest if not i in [0, num_replicas - 1]]
        my_source = bins[replica]
        if is_inner_replica and not (0 < my_source <= num_replicas - 1):
            print(
                'ERROR: bad interpolation source for replica' +
                ' {:d}: {:d}'.format(replica, my_source)
            )
            world.Abort()
        if group.rank == 0:
            y = {
                dest : np.empty(num_dof)
                for dest in my_dest
                if not dest == replica
            }
        else:
            y = {dest: None for dest in my_dest}
        if group.rank == 0:
            logfile.write(
                'Interpolation source:\t' + str(my_source) + '\n'
            )
            logfile.write(
                'Interpolation destinations:\t' + list_to_str(my_dest) + '\n'
            )
        world.Barrier()

        # Perform interpolations in the associated interval
        if (replica > 0):
            for dest in my_dest:
                x = ((target_parameterization[dest] - parameterization[replica - 1])
                    /(parameterization[replica] - parameterization[replica - 1])
                )
                y_chunk = (chunk_of_left_dof + x * chunk_dist_left)
                if dest == replica:
                    group.Gatherv(
                        [y_chunk, dof_chunk_sizes[group.rank], MPI.DOUBLE],
                        [my_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                        root=0
                    )
                else:
                    group.Gatherv(
                        [y_chunk, dof_chunk_sizes[group.rank], MPI.DOUBLE],
                        [y[dest], dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                        root=0
                    )
        world.Barrier()
        if group.rank == 0:
            logfile.write('Finished interpolations\n')
        if has_timer:
            timer.stamp()
            logfile.write(
                'Time since last timer call:\t' + str(timer.elapsed_time) + '\n'
            )

        # Communicate interpolated values to the replicas which will use them
        # Must only receive and send if my_source != replica
        if replica in my_dest:
            my_dest.remove(replica)
        if (group.rank == 0 and replica > 0):
            requests = []
            for i, dest in enumerate(my_dest):
                my_request = world.Isend(
                    buf=[y[dest], num_atoms, MPI.DOUBLE],
                    dest=group_roots[dest], tag=group_roots[dest]
                )
                requests.append(my_request)
            if is_inner_replica and (my_source != replica):
                my_request = world.Irecv(
                    buf=[my_dof, num_atoms, MPI.DOUBLE],
                    source=group_roots[my_source], tag=world.rank
                )
                requests.append(my_request)
            re = MPI.Request.Waitall(requests)

        # Reset coordinates  of atoms with  fixed dofs. This is  necessary even
        # though masked chunks have been used throughout because Gatherv copies
        # values  (it  ignores  masks).  So  my_dof[fixed_dofs]  will
        # contain whatever was in the corresponding chunks at these positions.
        my_dof[fixed_dofs] = my_dof_old[fixed_dofs]
        group.Scatterv(
            [my_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            chunk_of_my_dof, root=0)
        group.Scatterv(
            [my_dof_old, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            chunk_of_my_dof_old, root=0)

        # Compute displacement of this replica
        (_, my_displacement) = calc_chunk_distance(
                chunk_of_my_dof,
                chunk_of_my_dof_old,
                group
        )
        # Find maximum displacement
        displacements = np.empty(world.size)
        world.Allgather(
                sendbuf=[my_displacement, MPI.DOUBLE],
                recvbuf=[displacements, MPI.DOUBLE]
        )
        displacements = displacements[group_roots]
        max_displacement = displacements.max()
        if group.rank == 0:
            logfile.write('Maximum displacement:\t{:.3e}\n'.format(
                max_displacement)
            )
        if max_displacement < string_displacement_threshold:
            converged = True
            if group.rank == 0:
                logfile.write('Converged\n')
        my_dof_old = np.copy(my_dof)

        # Re-apply  periodic   boundary  conditions   Note:  we   re-apply  the
        # conditions according to the  initial state, before the interpolation.
        # If an atom has moved out of the box during interpolation, this should
        # not be  a problem, because we  perform only 1-step runs  with pre=yes
        # (default), so  pbcs are re-applied.  It might  even be valid  to skip
        # this step.
        chunk_of_my_dof = apply_pbc(
            chunk_of_my_dof, periodic_boundary, image_distances, 'wrap'
        )
        group.Allgatherv(
            sendbuf=[chunk_of_my_dof, dof_chunk_sizes[group.rank], MPI.DOUBLE],
            recvbuf=[my_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE]
        )
        my_lammps.scatter_atoms(
            "x", 1, 3, np.ctypeslib.as_ctypes(my_dof)
        )
        if world.rank == 0:
            save_reaction_path(
                iteration, length, parameterization, energies, displacements,
                tangential_forces, perpendicular_forces
            )
        world.Barrier()
        if converged: break

    world.Barrier()
    # Write output data
    my_lammps.command(
            'dump 1 all custom 1'
            + ' dump.replica_{:d} id type x y z c_PE_atom'.format(replica))
    my_lammps.command('dump_modify 1 pad 6'
            + ' format "%.7d %d %22.14e %22.14e %22.14e %22.14e"')
    my_lammps.command('run 0 post no')
    my_lammps.close()

def calc_chunk_distance(chunk1, chunk2, comm):
    """Calculate the distance between two chunks of data.

    Args:
        chunk1, chunk2 (ndarray): two chunks of data with the same dimension
        comm (MPI_COMM): communicator for reduction
    Returns:
        chunk_distance (ndarray): distance between chunk1 and chunk2
        norm_of_chunk_distance (float): Euclidean norm of chunk_distance
    """
    chunk_distance = chunk1 - chunk2
    squares = chunk_distance * chunk_distance
    norm_of_chunk_distance = np.empty(1)
    comm.Allreduce(
        sendbuf=[np.array(squares.sum()), MPI.DOUBLE],
        recvbuf=[norm_of_chunk_distance, MPI.DOUBLE],
        op=MPI.SUM)
    norm_of_chunk_distance = np.sqrt(norm_of_chunk_distance)
    return (chunk_distance, norm_of_chunk_distance)

def calc_image_distances(img_chunk, periodic_boundary, box_size):
    """Calculate the lengths for wrapping or unwrapping atomic coordinates
    across periodic boundaries.

    Args:
        img_chunk (ndarray): image flags
        periodic_boundary (list): element i = True if boundary is periodic in
            i-direction.
        box_size (ndarray): simulation box size
    Returns:
        image_distances (list, len=3): distance that must be subtracted
            from x, y, or z-coordinates to wrap them across the periodic
            boundaries. None if boundary is not periodic.
    Warning:
        - Triclinic boxes are not fully supported. In this case, only the
          non-inclinced direction can be periodic.
        - imgmax = 512 and img2bits = 20 only if Lammps has been compiled
          with LAMMPS_SMALLBIG
    """
    # Bit mask values for decoding Lammps image flags:
    imgmask = 1023
    imgmax = 512
    img2bits = 20
    image_distances = [None] * 3
    if periodic_boundary[0]:
        image_distances[0]  = np.bitwise_and(img_chunk, imgmask)
        image_distances[0] -= imgmax
        image_distances[0] *= box_size[0]
    if periodic_boundary[1]:
        image_distances[1]  = np.right_shift(img_chunk, img2bits)
        image_distances[1] &= imgmask
        image_distances[1] -= imgmax
        image_distances[1] *= box_size[1]
    if periodic_boundary[2]:
        image_distances[2]  = np.right_shift(img_chunk, img2bits)
        image_distances[2] -= imgmax
        image_distances[2] *= box_size[2]
    return image_distances

def apply_pbc(dof, periodic_boundary, image_distances, mode):
    """Apply periodic boundary conditions.

    Args:
        dof_chunk (ndarray): atomic coordinates
        periodic_boundary (list): element i = True if boundary in
            direction i is periodic
        image_distances (list, len=3): distance that must be subtracted
            from x, y, or z-coordinates to wrap them across the periodic
            boundaries. None if boundary is not periodic.
        mode (string): 'wrap' to wrap coordinates, 'unwrap' to unwrap
    Returns:
        dof_chunk (ndarray): atomic coordinates with pbc applied
    Warning:
        Triclinic boxes are not fully supported. In this case, only the
        non-inclinced direction can be periodic.
    """
    mode = str(mode)
    directions = range(3)
    if mode == 'unwrap':
        for i in directions:
            if periodic_boundary[i]:
                dof[i::3] += image_distances[i]
    elif mode == 'wrap':
        for i in directions:
            if periodic_boundary[i]:
                dof[i::3] -= image_distances[i]
    else:
        raise ValueError('Wrong mode: {:s}'.format(mode))
    return dof

def save_reaction_path(
        iteration, length, parameterization, energies, displacements,
        tangential_forces, perpendicular_forces):
    """Save reaction path data to disk.

    Args:
        length (float): length of the string
        parameterization (ndarray): current string parameterization
        energies (ndarray): energies of replicas
        displacements (ndarray): norms of displacements of each replica
    """
    if iteration == 0:
        with open('length.txt', 'w') as file:
            file.write('#Length of string\n')
        with open('parameterization.txt', 'w') as file:
            file.write('#Position of replicas 1,2...N\n')
        with open('energies.txt', 'w') as file:
            file.write('#Energy of replicas 1,2,...N\n')
        with open('displacements.txt', 'w') as file:
            file.write('#Displacement of replica 1,2,...N\n')
        with open('tangential_forces.txt', 'w') as file:
            file.write('#Tangential force of replica 1,2,...N\n')
        with open('perpendicular_forces.txt', 'w') as file:
            file.write('#Perpendicular force of replica 1,2,...N\n')
    with open('length.txt', 'a') as file:
        file.write('{:.8e}'.format(length) + '\n')
    with open('parameterization.txt', 'a') as file:
        line = vector_to_str(parameterization)
        file.write(line)
    with open('energies.txt', 'a') as file:
        line = vector_to_str(energies)
        file.write(line)
    with open('displacements.txt', 'a') as file:
        line = vector_to_str(displacements)
        file.write(line)
    with open('tangential_forces.txt', 'a') as file:
        line = vector_to_str(tangential_forces)
        file.write(line)
    with open('perpendicular_forces.txt', 'a') as file:
        line = vector_to_str(perpendicular_forces)
        file.write(line)

def vector_to_str(my_array, delimiter=' ', float_format='{:.8e}'):
    """Format the elements of a vector and return a string"""
    line = list(map(float_format.format, np.ravel(my_array)))
    line = delimiter.join(line)
    line += '\n'
    return line

def list_to_str(my_list, delimiter=" "):
    """Print a string with the list elements

    Args:
        my_list (list)
        delimiter (str): delimiter of list elements in string
    """
    return delimiter.join(map(str, my_list))

def safe_template_string(multiline_string):

    """ Put multiline lammps commands on a single line."""

    safe_template_string = ""
    for line in multiline_string.splitlines():
        if line.rstrip().endswith("&"):
            safe_template_string += line.rstrip().rstrip("&")
        else:
            safe_template_string += line + "\n"
    return CustomTemplate(safe_template_string)

class CustomTemplate(Template):
        delimiter = "?"

class Timer():

    """Timer for MPI"""

    def __init__(self):
        self.wtime = MPI.Wtime()
        self.elapsed_time = 0.0
    def stamp(self):
        current_time = MPI.Wtime()
        self.elapsed_time = current_time - self.wtime
        self.wtime = current_time

if __name__ == '__main__':
    main()
