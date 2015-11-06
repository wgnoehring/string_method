#!/bin/env python
"""Compute a reaction path using the modified string method.

How to run
----------
On deneb:
srun <script name>

Requirements
------------
- mpi4py version 1.3.1 or more recent
 (implies version 0.2.2. or more recent of ctypes)

Synopsis
--------
This is a  prototype implementation of the modified  string method with
parallelization on the replica level.
Weinan, Weiqing, Vanden-Eijnden,
The Journal of Chemical Physics 126, 164103 2007

Actually, this is a modification of the modified method.
Here, we don't interpolate the replica coordinates on
every timestep, but every num_lammps_steps timesteps.
This is less computationally intensive but still pushes
the replicas towards the transition path (or so I hope).

You'll see that the string moves very slowly after a while.
This is because the velocities are zeroed at interpolation steps.
I am working on a version where the velocities are kept (actually
I have a version, but samples are still exploding ...).

Input and Controls
------------------
The main parameters are presently hard-coded, right at the beginning
of the script. The Lammps instructions are passed via the template
string setup_template. Modify this template according to your needs.
You can perform subsitutions using the placeholder '?'.
Currently, the starting coordinates for the replicas are read from
a set of files with the name pattern <i>.atom, where i is the replica
number (zero offset). These must be Lammps dump files, but you can
change that easily by modifying the template.

WARNING
-------
This is a development version. I have had no time yet to write a
proper documentation, so see the comments in the code.
Also, the code still needs to be cleaned up and optimized.

Author
------
Wolfram Georg Noehring, Tue Jul 14 22:46:59 CEST 2015
"""

from mpi4py import MPI
import numpy as np
from lammps import lammps
import ctypes
from string import Template

# Todo: remove placeholders
# Todo: check where barriers are necessary
# Time estimate
#>> (3e-5*50000*311040/2)/3600
#ans =  64.800


def main():
    # Options:
    num_replicas = 16       # Number of replicas
    if (num_replicas < 3):
        raise ValueError('need at least three replicas')
    num_lammps_steps = 100  # Number of Lammps steps between interpolations
    max_num_steps =  200   # Maximum number of interpolations (so the total amount of lammps steps is max_num_steps * num_lammps_steps)
    TOL = 1.0e-4            # Stop criterion (max. string displacement)
    num_atoms = 311040      # Number of atoms
    dof = num_atoms * 3     # Degrees of freedom
    # Bit mask values for decoding Lammps image flags:
    imgmask = 1023
    imgmax = 512
    img2bits = 20
    periodic_x = 0
    periodic_y = 0
    periodic_z = 1

    # Split communicator into groups. Each group will simulate one replica.
    world = MPI.COMM_WORLD
    if world.size < num_replicas - 1:
        raise ValueError('Number of processes must be larger than'
            + ' number of replicas!')

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

    # Prepare for ring-like topology
    rneighbor_color = np.roll(range(num_replicas), -1)[color]
    lneighbor_color = np.roll(range(num_replicas), +1)[color]

    # Set up a number of lammps instances
    if MPI._sizeof(group) == ctypes.sizeof(ctypes.c_int):
        MPI_Comm = ctypes.c_int
    else:
        MPI_Comm = ctypes.c_void_p
    comm_ptr = MPI._addressof(group)
    comm_val = MPI_Comm.from_address(comm_ptr)
    this_lammps = lammps(name="", cmdargs="", ptr=comm_val)

    with open("in.lammps.string") as myfile:
        lmp_s_cmd=myfile.read()

    # Initialize all instances.
    # Note: we need an atom map for scatter_atoms.
    setup_template = safe_template_string(lmp_s_cmd)

    setup = setup_template.substitute(c = color)

    #lmp_s_cmd_check = "in.start_std." + `color`;
    #with open(lmp_s_cmd_check, "w") as myfile:
    #    myfile.write("{}".format(setup))

    # Initialize all replicas
    for line in setup.splitlines():
        this_lammps.command(line)

    # Define approx. evenly-sized chunks of 3*N-dimensional vector
    # Align such that dof belonging to an atom are all in one bin.
    bin_size = np.floor(dof / group.size).astype(int)
    bin_size -= bin_size%3
    chunk_bounds = [(i * bin_size, (i + 1) * bin_size)
        for i in range(group.size - 1)]
    chunk_bounds += [((group.size - 1) * bin_size, dof)]
    # Displacements for MPI vector scattering:
    chunk_displ = [c[0] for c in chunk_bounds]
    chunk_sizes = [len(range(c[0], c[1])) for c in chunk_bounds]
    # Chunk of atomic coordinates of this replica, and of left neighbor:
    my_chunk = np.empty(chunk_sizes[group.rank], dtype=float)
    if color > 0:
        nb_chunk = np.empty(chunk_sizes[group.rank], dtype=float)
    chunk_start = chunk_bounds[group.rank][0]
    chunk_end = chunk_bounds[group.rank][1]
    # Chunk for holding image flags:
    img_bin_size = bin_size / 3
    img_chunk_bounds = [(i * img_bin_size, (i + 1) * img_bin_size)
        for i in range(group.size - 1)]
    img_chunk_bounds += [((group.size - 1) * img_bin_size, num_atoms)]
    img_chunk_displ = [c[0] for c in img_chunk_bounds]
    img_chunk_sizes = [len(range(c[0], c[1])) for c in img_chunk_bounds]
    img_chunk = np.empty(img_chunk_sizes[group.rank], dtype=np.int32)

    # Extract cell size
    boxxlo = this_lammps.extract_global("boxxlo", 1)
    boxxhi = this_lammps.extract_global("boxxhi", 1)
    boxylo = this_lammps.extract_global("boxylo", 1)
    boxyhi = this_lammps.extract_global("boxyhi", 1)
    boxzlo = this_lammps.extract_global("boxzlo", 1)
    boxzhi = this_lammps.extract_global("boxzhi", 1)
    xprd = boxxhi - boxxlo
    yprd = boxyhi - boxylo
    zprd = boxzhi - boxzlo

    # Initialize old coordinates and unwrap coordinates
    this_replica_atom_coords_old = np.asarray(
        this_lammps.gather_atoms("x", 1, 3))
    this_replica_image_flags = np.asarray(
        this_lammps.gather_atoms("image", 0, 1))
    group.Scatterv(
        [this_replica_atom_coords_old, chunk_sizes, chunk_displ, MPI.DOUBLE],
        my_chunk, root=0)
    group.Scatterv(
        [this_replica_image_flags, img_chunk_sizes, img_chunk_displ, MPI.INT],
        img_chunk, root=0)
    # Warning: triclinic boxes with more than one periodic boundary are not supported yet
    if periodic_x:
        my_chunk[0::3] += (xprd * (np.bitwise_and(img_chunk, imgmask) - imgmax))
    if periodic_y:
        my_chunk[1::3] += (yprd * (np.bitwise_and(np.right_shift(img_chunk, img2bits), imgmask) - imgmax))
    if periodic_z:
        my_chunk[2::3] += (zprd * (np.right_shift(img_chunk, img2bits) - imgmax))
    group.Gatherv([my_chunk, chunk_sizes[group.rank], MPI.DOUBLE],
        [this_replica_atom_coords_old, chunk_sizes, chunk_displ, MPI.DOUBLE], root=0)

    lneighb_replica_energy = np.empty(1, dtype=float)
    lneighb_replica_atom_coords = np.empty(dof, dtype=float)

    if world.rank == 0:
        string_coords_history = np.zeros((max_num_steps, num_replicas), dtype=float)
        string_energy_history = np.zeros((max_num_steps, num_replicas), dtype=float)
        string_displ_history = np.zeros((max_num_steps, num_replicas), dtype=float)

    this_replica_atom_types = np.asarray(
        this_lammps.gather_atoms("type", 0, 1))
    this_replica_atom_types = np.vstack([this_replica_atom_types] * 3).reshape((-1,),order='F')
    fixed_atoms = np.where(np.logical_or(
                this_replica_atom_types == 3, this_replica_atom_types == 4))[0]

    converged = False
    world.Barrier()
    for step in range(max_num_steps):
        # Time-integrate each replica
        this_lammps.command("minimize 0 1e-8 {:d} {:d}".format(num_lammps_steps, num_lammps_steps))
        this_lammps.command("velocity all set 0.0 0.0 0.0")
        # Extract coordinates and energy (all processes in one group have the same view)
        # Note: np.asarray does, ideally, not involve copying!
        this_replica_atom_coords = np.asarray(
            this_lammps.gather_atoms("x", 1, 3))
        # Note: the z image flag can be obtained by right-shifting the image flag by img2bits
        # and subtracting imgmax. By default (lammps is compiled with the environment variable
        # LAMMPS_SMALLBIG), img2bits = 20 and imgmax = 512.
        this_replica_image_flags = np.asarray(
            this_lammps.gather_atoms("image", 0, 1))
        this_replica_energy = np.asarray(
            this_lammps.extract_variable("PE", "all", 0), dtype=float)
        # Apply image flags
        group.Scatterv(
            [this_replica_atom_coords, chunk_sizes, chunk_displ, MPI.DOUBLE],
            my_chunk, root=0)
        group.Scatterv(
            [this_replica_image_flags, img_chunk_sizes, img_chunk_displ, MPI.INT],
            img_chunk, root=0)
        # Warning: triclinic boxes with more than one periodic boundary are not supported yet
        if periodic_x:
            img_x = np.bitwise_and(img_chunk, imgmask)
            img_x -= imgmax
            img_x *= xprd
            my_chunk[0::3] += img_x
        if periodic_y:
            img_y = np.right_shift(img_chunk, img2bits)
            img_y &= imgmask
            img_y -= imgmax
            img_y *= yprd
            my_chunk[1::3] += img_y
        if periodic_z:
            img_z = np.right_shift(img_chunk, img2bits)
            img_z -= imgmax
            img_z *= zprd
            my_chunk[2::3] += img_z
        group.Gatherv([my_chunk, chunk_sizes[group.rank], MPI.DOUBLE],
            [this_replica_atom_coords, chunk_sizes, chunk_displ, MPI.DOUBLE], root=0)
        world.Barrier()
        #print('#Color {:d} finished minimization and scattered variables'.format(color))

        # Send coordinates and energy to the right neighbor
        if (group.rank == 0):
            source = group_roots[lneighbor_color]
            dest = group_roots[rneighbor_color]
            world.Sendrecv([this_replica_atom_coords, MPI.DOUBLE], dest=dest,
                recvbuf=[lneighb_replica_atom_coords, MPI.DOUBLE], source=source)
            world.Sendrecv([this_replica_energy, MPI.DOUBLE], dest=dest,
                recvbuf=[lneighb_replica_energy, MPI.DOUBLE], source=source)

        # Scatter coordinates to neighbors and compute norm of difference
        if (color > 0):
            group.Scatterv(
                [lneighb_replica_atom_coords, chunk_sizes, chunk_displ, MPI.DOUBLE],
                nb_chunk, root=0)
            group.Scatterv(
                [this_replica_atom_coords, chunk_sizes, chunk_displ, MPI.DOUBLE],
                my_chunk, root=0)
            # This difference will be recycled later during the interpolation:
            my_chunk -= nb_chunk
            squares = my_chunk * my_chunk
            sendbuf = np.array(squares.sum(), dtype=float)
            lneighb_dist = np.empty(1, dtype=float)
            group.Allreduce([sendbuf, MPI.DOUBLE],
                [lneighb_dist, MPI.DOUBLE], op=MPI.SUM)
            lneighb_dist = np.sqrt(lneighb_dist)
        else:
            lneighb_dist = 0.0

        # Here, we would weight by energy. The factor 1.0 is a placeholder.
        if (color > 0 and group.rank == 0):
            length_increment = 1.0 * lneighb_dist
        lneighb_dist = np.asarray(lneighb_dist, dtype=float)
        world.Barrier()
        #print('#Color {:d} computed neighbor distance'.format(color))

        # Determine the positions of the replicas on the string
        old_string_coord = np.empty(world.size, dtype=float)
        world.Allgather([lneighb_dist, MPI.DOUBLE], [old_string_coord, MPI.DOUBLE])
        old_string_coord = old_string_coord[group_roots]
        old_string_coord = np.cumsum(old_string_coord, out=old_string_coord)
        old_string_coord /= old_string_coord[-1]
        # Again, here we would have to do energy-weighting
        sendbuf = np.array(float(color + 1) / float(num_replicas), dtype=float)
        new_string_coord = np.empty(world.size, dtype=float)
        world.Allgather([sendbuf, MPI.DOUBLE], [new_string_coord, MPI.DOUBLE])
        new_string_coord = new_string_coord[group_roots]

        # Determine which replica is associated with the interval
        # in which the new position lies and request interpolation
        bins = np.searchsorted(old_string_coord, new_string_coord, 'left')
        this_replica_dest = np.where(bins == color)[0]
        this_replica_dest = [i for i in this_replica_dest
            if not i in [0, num_replicas - 1]]
        this_replica_source = bins[color]
        if is_inner_replica and not (0 < this_replica_source <= num_replicas - 1):
            print('ERROR: bad interpolation source for replica {:d}: {:d}'.format(
                color, this_replica_source))
            world.Abort()
        if group.rank == 0:
            y = {dest : np.empty(dof, dtype=float) for dest in this_replica_dest
                if not dest == color}
        else:
            y = {dest: None for dest in this_replica_dest}
        world.Barrier()
        #print('#Color {:d} requested interpolations'.format(color))

        # Perform interpolations in the associated interval
        if (color > 0):
            for dest in this_replica_dest:
                x = new_string_coord[dest]
                y_chunk = (nb_chunk + my_chunk
                    * ((x - old_string_coord[color - 1])
                    / (old_string_coord[color] - old_string_coord[color - 1])))
                if dest == color:
                    group.Gatherv([y_chunk, chunk_sizes[group.rank], MPI.DOUBLE],
                        [this_replica_atom_coords, chunk_sizes, chunk_displ, MPI.DOUBLE], root=0)
                else:
                    group.Gatherv([y_chunk, chunk_sizes[group.rank], MPI.DOUBLE],
                        [y[dest], chunk_sizes, chunk_displ, MPI.DOUBLE], root=0)
        #print('#Color {:d} performed interpolations'.format(color))
        #print('#Color {:d} my source: {:d}'.format(color, this_replica_source))
        print('#Color {:d} my dests'.format(color), this_replica_dest)
        #print('#Color {:d} bins:', bins[color])
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
                    buf=[this_replica_atom_coords, num_atoms, MPI.DOUBLE],
                    source=group_roots[this_replica_source], tag=world.rank))
            re = MPI.Request.Waitall(requests)
            #print('#Color {:d} waitall returns:', re)

        #if this_replica_source != color and is_inner_replica and group.rank == 0:
        #    print('#Color {:d} setting up my receive'.format(color))
        #    world.Recv([this_replica_atom_coords, MPI.DOUBLE],
        #        source=group_roots[this_replica_source])
        #world.Barrier()
        #if (color > 0 and group.rank == 0):
        #    for dest in this_replica_dest:
        #        world.Rsend([y[dest], MPI.DOUBLE],
        #            dest=group_roots[dest])
        #print('#Color {:d} setup sends'.format(color))

        #group.Bcast([this_replica_atom_coords, MPI.DOUBLE], root=0)
        #my_chunk = this_replica_atom_coords[chunk_start:chunk_end]

        # Reset type 2 atoms
        this_replica_atom_coords[fixed_atoms] = this_replica_atom_coords_old[fixed_atoms]
        group.Scatterv(
            [this_replica_atom_coords, chunk_sizes, chunk_displ, MPI.DOUBLE],
            my_chunk, root=0)
        my_chunk_old = np.empty(chunk_sizes[group.rank], dtype=float)
        group.Scatterv(
            [this_replica_atom_coords_old, chunk_sizes, chunk_displ, MPI.DOUBLE],
            my_chunk_old, root=0)

        # Compute displacement of this replica
        squares = (my_chunk - my_chunk_old)
        squares *= squares
        squares = squares.sum()
        sendbuf = np.array(squares, dtype=float)
        print('#ID, squares {:d} {:d} {:.8f}'.format(color, group.rank, squares))
        my_displ = np.empty(1, dtype=float)
        group.Allreduce([sendbuf, MPI.DOUBLE],
            [my_displ, MPI.DOUBLE], op=MPI.SUM)
        my_displ = np.asarray(np.sqrt(my_displ), dtype=float)
        this_replica_atom_coords_old = np.copy(this_replica_atom_coords)
        world.Barrier()

        # Re-apply periodic boundary conditions
        # Note: we re-apply the conditions according to the initial state,
        # before the interpolation. If an atom has moved out of the box
        # during interpolation, this should not be a problem, because
        # we perform only 1-step runs with pre=yes (default), so pbcs
        # are re-applied. It might even be valid to skip this step.
        if periodic_x:
            my_chunk[0::3] -= img_x
        if periodic_y:
            my_chunk[1::3] -= img_y
        if periodic_z:
            my_chunk[2::3] -= img_z
        group.Allgatherv([my_chunk, chunk_sizes[group.rank], MPI.DOUBLE],
            [this_replica_atom_coords, chunk_sizes, chunk_displ, MPI.DOUBLE])

        # Find maximum displacement
        all_displ = np.empty(world.size, dtype=float)
        world.Allgather([my_displ, MPI.DOUBLE], [all_displ, MPI.DOUBLE])
        all_displ = all_displ[group_roots]
        if all_displ.max() < TOL:
            converged = True
        #if is_inner_replica:
        this_lammps.scatter_atoms("x", 1, 3, np.ctypeslib.as_ctypes(
            this_replica_atom_coords))
        world.Barrier()
        all_energies = np.empty(world.size, dtype=float)
        world.Allgather([this_replica_energy, MPI.DOUBLE], [all_energies, MPI.DOUBLE])
        all_energies = all_energies[group_roots]
        if world.rank == 0:
            string_coords_history[step, :] = old_string_coord
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
