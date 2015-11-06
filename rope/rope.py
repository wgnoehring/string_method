#!/bin/env python
"""Calculate a transition path with the modified string method.

Usage
-----
On deneb:
srun <script name>

Requirements
------------
mpi4py version 1.3.1 or more recent
(implies version 0.2.2. or more recent of ctypes)

Synopsis
--------
This is a prototype implementation of the modified string method, see Weinan et
al. The Journal of Chemical Physics 126, 164103 2007.

The general workflow is as follows:
    - First, MPI  communicators are set up and a  number of num_replicas Lammps
    instances are created  and initialized. Note that more than  one CPU can be
    associated with each Lammps instance, i.e. it is possible to parallelize on
    the replica level. - Second, the transition path is calculated iteratively.
    The maximum  number of iterations  is max_num_steps.  As the first  step in
    each iteration, the equations of motion are integrated with Lammps. In each
    iteration, num_lammps_steps timesteps are simulated. After integration, the
    atomic  positions are  extracted and  the string  is reparameterized  using
    linear interpolation.

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
from lammps import lammps
from string import Template

def main():
    # Options:
    num_replicas = 16       # Number of replicas
    if (num_replicas < 3):
        raise ValueError('need at least three replicas')
    num_lammps_steps = 100  # Number of Lammps steps between interpolations
    max_num_steps =  200    # Maximum number of reparameterizations
    TOL = 1.0e-4            # Stop criterion (max. string displacement)
    num_atoms = 311040      # Number of atoms
    num_dof = num_atoms * 3 # Degrees of freedom
    periodic_x = 0
    periodic_y = 0
    periodic_z = 1
    # Bit mask values for decoding Lammps image flags:
    imgmask = 1023
    imgmax = 512
    img2bits = 20

    # Split communicator into groups. Each group will simulate one replica.
    world = MPI.COMM_WORLD
    if world.size < num_replicas - 1:
        raise ValueError(
                'Number of processes must be larger than number of replicas!'
        )
    replica_association = np.array_split(np.arange(world.size), num_replicas)
    for i, ranks in enumerate(replica_association):
        if world.rank in ranks:
            color = i
            break
    del(replica_association)
    group = world.Split(color=color, key=world.rank)
    is_inner_replica = int(0 < color < num_replicas - 1)

    # Get the ranks of the group-roots
    sendbuf = np.zeros(1, dtype=np.int)
    if group.rank == 0:
        sendbuf[0] = np.int(world.rank)
    else:
        sendbuf[0] = np.int(-1.0)
    recvbuf = np.zeros(world.size, dtype=np.int)
    world.Allgather([sendbuf, MPI.INT], [recvbuf, MPI.INT])
    group_roots = np.empty(num_replicas, dtype=np.int)
    if world.rank == 0:
        group_roots = np.asarray([i for i in recvbuf if i>-1], dtype=np.int)
    world.Bcast([group_roots, MPI.INT], root=0)

    # Determine left and right neighbors in a ring-like MPI topology
    left_replica_color = np.roll(range(num_replicas), +1)[color]
    right_replica_color = np.roll(range(num_replicas), -1)[color]
    left_replica_root = group_roots[left_replica_color]
    right_replica_root = group_roots[right_replica_color]
    # Declare arrays for neighbor data
    left_replica_energy = np.empty(1)
    right_replica_energy = np.empty(1)
    left_replica_dof = np.empty(num_dof)
    right_replica_dof = np.empty(num_dof)

    # Create Lammps instances
    if MPI._sizeof(group) == ctypes.sizeof(ctypes.c_int):
        MPI_Comm = ctypes.c_int
    else:
        MPI_Comm = ctypes.c_void_p
    comm_ptr = MPI._addressof(group)
    comm_val = MPI_Comm.from_address(comm_ptr)
    this_lammps = lammps(name="", cmdargs="", ptr=comm_val)

    # Initialize replicas
    # Note: scatter_atoms requires an atom map
    with open("in.lammps.string") as myfile:
        lammps_setup_file = myfile.read()
    setup_template = safe_template_string(lammps_setup_file)
    setup = setup_template.substitute(c = color)
    for line in setup.splitlines():
        this_lammps.command(line)

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
    chunk_of_this_replica_dof = np.empty(dof_chunk_sizes[group.rank])
    chunk_of_this_replica_force = np.empty(dof_chunk_sizes[group.rank])
    if color > 0:
        chunk_of_left_replica_dof = np.empty(dof_chunk_sizes[group.rank])
        chunk_of_right_replica_dof = np.empty(dof_chunk_sizes[group.rank])
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

    # Extract simulation box size
    box_size = np.zeros(3)
    box_size[0] += this_lammps.extract_global("boxxhi", 1)
    box_size[0] -= this_lammps.extract_global("boxxlo", 1)
    box_size[1] += this_lammps.extract_global("boxyhi", 1)
    box_size[1] -= this_lammps.extract_global("boxylo", 1)
    box_size[2] += this_lammps.extract_global("boxzhi", 1)
    box_size[2] -= this_lammps.extract_global("boxzlo", 1)

    # Initialize old coordinates and unwrap
    this_replica_dof_old = np.asarray(
        this_lammps.gather_atoms("x", 1, 3)
    )
    this_replica_image_flags = np.asarray(
        this_lammps.gather_atoms("image", 0, 1)
    )
    group.Scatterv(
        [this_replica_dof_old, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
        chunk_of_this_replica_dof, root=0
    )
    group.Scatterv(
        [this_replica_image_flags, img_chunk_sizes, img_chunk_displ, MPI.INT],
        img_chunk, root=0
    )
    image_distances = calc_image_distances(
        img_chunk, periodic_boundary, box_size
    )
    chunk_of_this_replica_dof = apply_pbc(
        chunk_of_this_replica_dof, periodic_boundary, image_distances, 'unwrap'
    )
    group.Gatherv(
        [chunk_of_this_replica_dof, dof_chunk_sizes[group.rank], MPI.DOUBLE],
        [this_replica_dof_old, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
        root=0
    )
    this_replica_atom_types = np.asarray(
        this_lammps.gather_atoms("type", 0, 1)
    )
    # Todo: mask arrays
    this_replica_atom_types = np.vstack([this_replica_atom_types] * 3)
    this_replica_atom_types = this_replica_atom_types.reshape((-1,),order='F')
    fixed_atoms = np.where(np.logical_or(
                this_replica_atom_types == 3, this_replica_atom_types == 4))[0]

    # Rank 0 stores the convergence history
    if world.rank == 0:
        string_coords_history = np.zeros((max_num_steps, num_replicas))
        string_energy_history = np.zeros((max_num_steps, num_replicas))
        string_displ_history = np.zeros((max_num_steps, num_replicas))
    converged = False
    world.Barrier()
    minimize_command = "minimize 0 1e-8 {:d} {:d}".format(
            num_lammps_steps, num_lammps_steps
    )
    for step in range(max_num_steps):
        # Evolve replica
        this_lammps.command(minimize_command)
        this_lammps.command("velocity all set 0.0 0.0 0.0")
        # Extract coordinates and energy
        # Note: - all processes in the group have the same view
        #       - np.asarray from ctypes does not copy, ideally
        this_replica_dof = np.asarray(
            this_lammps.gather_atoms("x", 1, 3)
        )
        this_replica_image_flags = np.asarray(
            this_lammps.gather_atoms("image", 0, 1)
        )
        this_replica_force = np.asarray(
            this_lammps.gather_atoms("f", 1, 3)
        )
        this_replica_energy = np.asarray(
            this_lammps.extract_variable("PE", "all", 0)
        )
        # Apply image flags
        group.Scatterv(
            [this_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            chunk_of_this_replica_dof, root=0
        )
        group.Scatterv(
            [this_replica_image_flags, img_chunk_sizes, img_chunk_displ, MPI.INT],
            img_chunk, root=0
        )
        image_distances = calc_image_distances(
            img_chunk, periodic_boundary, box_size
        )
        chunk_of_this_replica_dof = apply_pbc(
            chunk_of_this_replica_dof, periodic_boundary, image_distances, 'unwrap'
        )
        group.Gatherv(
            [chunk_of_this_replica_dof, dof_chunk_sizes[group.rank], MPI.DOUBLE],
            [this_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            root=0
        )
        world.Barrier()

        # Get coordinates and energy of left and right neighbor
        if (group.rank == 0):
            world.Sendrecv(
                sendbuf=[this_replica_dof, MPI.DOUBLE],
                dest=right_replica_root,
                recvbuf=[left_replica_dof, MPI.DOUBLE],
                source=left_replica_root
            )
            world.Sendrecv(
                sendbuf=[this_replica_energy, MPI.DOUBLE],
                dest=right_replica_root,
                recvbuf=[left_replica_energy, MPI.DOUBLE],
                source=left_replica_root
            )
            world.Sendrecv(
                sendbuf=[this_replica_dof, MPI.DOUBLE],
                dest=left_replica_root,
                recvbuf=[right_replica_dof, MPI.DOUBLE],
                source=right_replica_root
            )
            world.Sendrecv(
                sendbuf=[this_replica_energy, MPI.DOUBLE],
                dest=left_replica_root,
                recvbuf=[right_replica_energy, MPI.DOUBLE],
                source=right_replica_root
            )

        # Compute distance to left neighbor
        if (color > 0):
            group.Scatterv(
                [left_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                chunk_of_left_replica_dof, root=0)
            group.Scatterv(
                [this_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                chunk_of_this_replica_dof, root=0)
            chunk_dist_left, norm_of_distance_to_left = calc_chunk_distance(
                    chunk_of_this_replica_dof, chunk_of_left_replica_dof, group)
        else:
            norm_of_distance_to_left = 0.0

        # Compute the tangent vector and the force perpendicular to the path
        if is_inner_replica:
            if   (left_replica_energy
                > this_replica_energy
                > right_replica_energy):
                chunk_of_tangent = chunk_dist_left / norm_of_distance_to_left
            elif (left_replica_energy
                < this_replica_energy
                < right_replica_energy):
                group.Scatterv(
                    [right_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                    chunk_of_right_replica_dof, root=0)
                chunk_dist, norm_of_distance = calc_chunk_distance(
                        chunk_of_right_replica_dof,
                        chunk_of_this_replica_dof, group)
                chunk_of_tangent = chunk_dist / norm_of_distance
            else:
                group.Scatterv(
                    [right_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                    chunk_of_right_replica_dof, root=0)
                chunk_dist, norm_of_distance = calc_chunk_distance(
                        chunk_of_right_replica_dof, chunk_of_left_replica_dof, group)
                chunk_of_tangent = chunk_dist / norm_of_distance
            group.Scatterv(
                [this_replica_force, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                chunk_of_this_replica_force, root=0)
            sendbuf = np.vdot(chunk_of_this_replica_force, chunk_of_tangent)
            projection = np.empty(1)
            group.Allreduce([sendbuf, MPI.DOUBLE],
                [projection, MPI.DOUBLE], op=MPI.SUM)
            chunk_of_perp_force, norm_of_perp_force = calc_chunk_distance(
                    chunk_of_this_replica_force,
                    projection * chunk_of_tangent, group)

        # Compute total length of string
        # Here, we would weight by energy. The factor 1.0 is a placeholder.
        if (color > 0 and group.rank == 0):
            length_increment = 1.0 * norm_of_distance_to_left
        norm_of_distance_to_left = np.asarray(norm_of_distance_to_left)
        world.Barrier()

        # Determine the positions of the replicas on the string
        current_string_pos = np.empty(world.size)
        world.Allgather(
                [norm_of_distance_to_left, MPI.DOUBLE],
                [current_string_pos, MPI.DOUBLE])
        current_string_pos = current_string_pos[group_roots]
        current_string_pos = np.cumsum(current_string_pos, out=current_string_pos)
        current_string_pos /= current_string_pos[-1]
        # Calculate ideal replica positions.
        # If the current positions are energy-weighted, then then target
        # positions must be weighted, too.
        sendbuf = np.array(float(color + 1) / float(num_replicas))
        target_string_pos = np.empty(world.size)
        world.Allgather([sendbuf, MPI.DOUBLE], [target_string_pos, MPI.DOUBLE])
        target_string_pos = target_string_pos[group_roots]

        # Determine which replica is associated with the interval
        # in which the new position lies and request interpolation
        bins = np.searchsorted(current_string_pos, target_string_pos, 'left')
        this_replica_dest = np.where(bins == color)[0]
        this_replica_dest = [
            i for i in this_replica_dest if not i in [0, num_replicas - 1]
        ]
        this_replica_source = bins[color]
        if is_inner_replica and not (0 < this_replica_source <= num_replicas - 1):
            print(
                'ERROR: bad interpolation source for replica' +
                ' {:d}: {:d}'.format(color, this_replica_source)
            )
            world.Abort()
        if group.rank == 0:
            y = {
                dest : np.empty(num_dof)
                for dest in this_replica_dest
                if not dest == color
            }
        else:
            y = {dest: None for dest in this_replica_dest}
        world.Barrier()

        # Perform interpolations in the associated interval
        if (color > 0):
            for dest in this_replica_dest:
                x = ((target_string_pos[dest] - current_string_pos[color - 1])
                    /(current_string_pos[color] - current_string_pos[color - 1])
                )
                y_chunk = (chunk_of_left_replica_dof + x * chunk_dist_left)
                if dest == color:
                    group.Gatherv(
                        [y_chunk, dof_chunk_sizes[group.rank], MPI.DOUBLE],
                        [this_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                        root=0
                    )
                else:
                    group.Gatherv(
                        [y_chunk, dof_chunk_sizes[group.rank], MPI.DOUBLE],
                        [y[dest], dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
                        root=0
                    )
        print('#Color {:d} my dests'.format(color), this_replica_dest)
        world.Barrier()

        # Communicate interpolated values to the replicas which will use them
        # Must only receive and send if my_source != color
        if color in this_replica_dest:
            this_replica_dest.remove(color)
        if (group.rank == 0 and color > 0):
            requests = []
            for i, dest in enumerate(this_replica_dest):
                requests.append(world.Isend(
                    buf=[y[dest], num_atoms, MPI.DOUBLE],
                    dest=group_roots[dest], tag=group_roots[dest]))
            if is_inner_replica and (this_replica_source != color):
                requests.append(world.Irecv(
                    buf=[this_replica_dof, num_atoms, MPI.DOUBLE],
                    source=group_roots[this_replica_source], tag=world.rank))
            re = MPI.Request.Waitall(requests)
            #print('#Color {:d} waitall returns:', re)

        # Reset type 2 atoms
        this_replica_dof[fixed_atoms] = this_replica_dof_old[fixed_atoms]
        group.Scatterv(
            [this_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            chunk_of_this_replica_dof, root=0)
        chunk_of_this_replica_dof_old = np.empty(dof_chunk_sizes[group.rank])
        group.Scatterv(
            [this_replica_dof_old, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE],
            chunk_of_this_replica_dof_old, root=0)

        # Compute displacement of this replica
        (_, my_displ) = calc_chunk_distance(
                chunk_of_this_replica_dof,
                chunk_of_this_replica_dof_old
        )
        this_replica_dof_old = np.copy(this_replica_dof)
        world.Barrier()

        # Re-apply  periodic   boundary  conditions   Note:  we   re-apply  the
        # conditions according to the  initial state, before the interpolation.
        # If an atom has moved out of the box during interpolation, this should
        # not be  a problem, because we  perform only 1-step runs  with pre=yes
        # (default), so  pbcs are re-applied.  It might  even be valid  to skip
        # this step.
        chunk_of_this_replica_dof = apply_pbc(
            chunk_of_this_replica_dof, periodic_boundary,
            image_distances, 'wrap'
        )
        group.Allgatherv(
            [chunk_of_this_replica_dof, dof_chunk_sizes[group.rank], MPI.DOUBLE],
            [this_replica_dof, dof_chunk_sizes, dof_chunk_displ, MPI.DOUBLE]
        )

        # Find maximum displacement
        all_displ = np.empty(world.size)
        world.Allgather([my_displ, MPI.DOUBLE], [all_displ, MPI.DOUBLE])
        all_displ = all_displ[group_roots]
        if all_displ.max() < TOL:
            converged = True
        this_lammps.scatter_atoms(
            "x", 1, 3, np.ctypeslib.as_ctypes(this_replica_dof)
        )
        world.Barrier()
        all_energies = np.empty(world.size)
        world.Allgather(
                [this_replica_energy, MPI.DOUBLE],
                [all_energies, MPI.DOUBLE]
        )
        all_energies = all_energies[group_roots]
        if world.rank == 0:
            string_coords_history[step, :] = current_string_pos
            string_energy_history[step, :] = all_energies
            string_displ_history[step, :] = all_displ
        if world.rank == 0:
            print('Maximum displacement: ' + str(all_displ.max()))
            out =  ['{:.8f}'.format(energy) for energy in all_energies]
            print('#Replica energies:' + ' '.join(out))
        print('#Color / converged', color, converged)
        world.Barrier()
        if converged: break

    world.Barrier()
    # Write output data
    this_lammps.command('dump 1 all custom 1 dump.replica_{:d} id type x y z c_PE_atom'.format(color))
    this_lammps.command('dump_modify 1 pad 6 format "%.7d %d %22.14e %22.14e %22.14e %22.14e"')
    this_lammps.command('run 0 post no')
    this_lammps.close()
    if world.rank == 0:
        np.savetxt('string_coords_history.npy', string_coords_history[:step+1])
        np.savetxt('string_energy_history.npy', string_energy_history[:step+1])
        np.savetxt('string_displ_history.npy', string_displ_history[:step+1])

def calc_chunk_distance(chunk1, chunk2, comm)
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
    sendbuf = np.array(squares.sum())
    norm_of_chunk_distance = np.empty(1)
    comm.Allreduce([sendbuf, MPI.DOUBLE],
        [norm_of_chunk_distance, MPI.DOUBLE], op=MPI.SUM)
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
    if mode == 'unwrap':
        for i in xrange(3):
            dof[i::3] += image_distances[i]
    elif mode == 'wrap':
        for i in xrange(3):
            dof[i::3] -= image_distances[i]
    else:
        raise ValueError('Wrong mode: {:s}'.format(mode))
    return dof

def safe_template_string(multiline_string):
    """ Put multiline lammps commands on a single line.
    """
    safe_template_string = ""
    for line in multiline_string.splitlines():
        if line.rstrip().endswith("&"):
            safe_template_string += line.rstrip().rstrip("&")
        else:
            safe_template_string += line + "\n"
    return CustomTemplate(safe_template_string)

class CustomTemplate(Template):
        delimiter = "?"

if __name__ == '__main__':
    main()
