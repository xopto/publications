# -*- coding: utf-8 -*-
################################ Begin license #################################
# Copyright (C) Laboratory of Imaging technologies,
#               Faculty of Electrical Engineering,
#               University of Ljubljana.
#
# This file is part of PyXOpto.
#
# PyXOpto is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyXOpto is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyXOpto. If not, see <https://www.gnu.org/licenses/>.
################################# End license ##################################

import os
import os.path
import time
import argparse
import threading

from xopto.mcvox import mc
from xopto.cl import clinfo

import numpy as np


class McProgressMonitor:
    def __init__(self, mcsim, target=None, period: float = 0.5):
        self._mcsim = mcsim
        self._period = max(0.1, float(period))
        self._processed = 0
        self._threads = 0
        self._target = 0

        self._track = False
        self._stop = False
        self._condition = threading.Semaphore(0)

        self._thread = threading.Thread(target=self._proc, args=(mcsim,))
        self._thread.start()
        
        if target is not None:
            self.start(target)

    def start(self, target: int):
        self._target = int(target)
        self._processed = 0
        self._threads = 0
        self._track = True
        self._condition.release()

    def resume(self, target: int = None):
        if target is not None:
            self._target = int(target)

        self._track = True

    def progress(self) -> float:
        return min(self._processed/self._target, 1.0)

    def target(self) -> int:
        return self._target

    def processed(self) -> int:
        return min(self._target, self._processed)

    def threads(self) -> int:
        return self._threads

    def stop(self):
        self._track = False
        self._processed = self._target

    def terminate(self):
        self._stop = True
        self._track = False
        self._condition.release()

    def _proc(self, mcsim):
        num_processed = np.zeros([1], dtype=mcsim.types.np_cnt)
        num_threads = np.zeros([1], dtype=np.uint32)
        queue = mc.cl.CommandQueue(mcsim.cl_context)

        while not self._stop:
            #print('\nloop 1\n')
            self._condition.acquire()
            while self._track and self._target > self._processed:
                #print('\nloop 2\n')
                cl_num_packets = mcsim.cl_buffers.get('num_processed_packets')
                cl_num_kernels = mcsim.cl_buffers.get('num_kernels')
                if cl_num_packets is not None and cl_num_kernels is not None:
                    mc.cl.enqueue_copy(queue, num_processed, cl_num_packets)
                    mc.cl.enqueue_copy(queue, num_threads, cl_num_kernels)
                    self._threads = num_threads[0]
                    if self._processed != num_processed[0]:
                        self._processed = num_processed[0]
                        self.report()
                    if self._target <= self._processed:
                        print()
                        
                time.sleep(self._period)

    def report(self):
        #print('Progress: {:,d} kernels, {:,d}/{:,d} packets ({:3.0f}%)'.format(
        #    self.threads(), self.processed(), self.target(),
        #    100*self.progress()), end='\r')
        N = 42
        n = int(self.progress()*N)
        print('|{}>{}| {:d}%'.format(
            '-'*n, ' '*(N - n), int(100.0*self.progress())), end='\r')


if __name__ == '__main__':
    cl_name = None
    mc_options = []
    cl_build_options = ['-cl-fast-relaxed-math','-cl-mad-enable']
    material_mem_name = 'global'

    parser = argparse.ArgumentParser(description='MCVOX performance test')
    parser.add_argument('-d', '--device', metavar='DEVICE_NAME',
                        type=str, default='',
                        help='Unique part of the target OpenCL device name.')
    parser.add_argument('-i', '--index', metavar='DEVICE_INDEX',
                        type=int, default=0,
                        help='OpenCL device index. Use 0 for the first device (default).')
    parser.add_argument('-n', '--nphotons', metavar='NUM_PHOTONS',
                        type=float, default=2e9,
                        help='Number of photon packets (default is 2,000,000,0000).')
    parser.add_argument('-o', '--output', metavar='OUTPUT_FILENAME',
                        type=str, default='',
                        help='Custom output file name for the simulation results.')
    parser.add_argument('-m', '--method', metavar='MC_METHOD', type=str,
                        choices = ('aw', 'ar', 'mbl'),
                        default='aw', required=False,
                        help='Select a Monte Carlo simulation method. Use '
                             '"ar" for Albedo Rejection, '
                             '"aw" for Albedo Weight (default) or '
                             '"mbl" for Microscopic Beer-Lambert.')
    parser.add_argument('-t', '--threshold', metavar='LOTTERY_THRESHOLD',
                        type=float, default=1e-4,
                        help='Termination lottery threshold (default=1e-4).'
                             'A value of 0 will switch off the lottery')
    parser.add_argument('-c', '--constantmem', action='store_true',
                        default=False, required=False,
                        help='Moves the material data to constant memory. '
                             'Constant memory is fully cached and can '
                             'lead to significant performance gains, in particular '
                             'on older GPUs. Note that the amount of constant memory '
                             'on GPUs is typically limited to about 64 kB, which '
                             'can support roughly 2000 materials (some of the '
                             'constant memory is already in use by the MC kernel).')
    parser.add_argument('-e', '--disablecache', action='store_true',
                        default=False, required=False,
                        help='Disables the software cache of fluence accumulators. This '
                             'option reduce performance on platfors that are '
                             'performance-limited by the bandwidth of the global memory.'
                             'On platforms that are not performance-limited by the bandwidth '
                             'of the global memory, disabling the software cache will '
                             'likely lead to some performance gains.')

    args = parser.parse_args()
    
    if args.device:
        cl_name = str(args.device)
    cl_index = max(int(args.index), 0)
    num_packets = int(max(min(args.nphotons, 4e9), 1))
    output = str(args.output)
    kernel_method, kernel_method_name = {
        'ar': ('albedo_rejection', 'Albedo Rejection'),
        'aw': ('albedo_weight', 'Albedo Weight'),
        'mbl': ('microscopic_beer_lambert', 'Microscopic Beer-Lambert'),
    }.get(str(args.method))
    mc_options.append(getattr(mc.mcoptions.McMethod, kernel_method))
    lottery_threshold = args.threshold
    if lottery_threshold <= 0.0:
        mc_options.append(mc.mcoptions.McUseLottery.off)
    else:
        mc_options.append(mc.mcoptions.McMinimumPacketWeight(lottery_threshold))
    if args.constantmem:
        material_mem_name = 'constant'
        mc_options.append(mc.mcoptions.McMaterialMemory.constant_mem)
    if args.disablecache:
        mc_options.append(mc.mcoptions.McUseFluenceCache.off)
    else:
        mc_options.append(mc.mcoptions.McUseFluenceCache.on)

    # This example shows how to simulate deposited energy using the 
    # voxelized Monte Carlo method for a simple case of a two-layered 
    # skin model with an embedded blood vessel.

    cl_device = clinfo.device(cl_name, index=cl_index)

    # DEFINE VOXELS AND TISSUE STRUCTURE
    # number of bins and size
    nx = 21
    ny = 21
    nz = 20
    voxel_size = 5e-05

    # define each axis
    xaxis = mc.mcgeometry.Axis(
        start=-nx/2*voxel_size, 
        stop=nx/2*voxel_size, 
        n=nx
    )

    yaxis = mc.mcgeometry.Axis(
        start=-ny/2*voxel_size, 
        stop=ny/2*voxel_size, 
        n=ny
    )

    zaxis = mc.mcgeometry.Axis(
        start=0.0, 
        stop=nz*voxel_size,
        n=nz
    )

    # define voxels
    voxels = mc.mcgeometry.Voxels(
        xaxis=xaxis, 
        yaxis=yaxis, 
        zaxis=zaxis
    )

    # DEFINE MATERIALS
    # surrounding medium
    material_surrounding = mc.mcmaterial.Material(
        n=1.337,
        mua=0.01,
        mus=100.0,
        pf=mc.mcpf.Hg(1.0)
    )

    # epidermis
    material_epidermis = mc.mcmaterial.Material(
        n=1.337,
        mua=1657.24,
        mus=37593.98,
        pf=mc.mcpf.Hg(0.9)
    )

    # dermis
    material_dermis = mc.mcmaterial.Material(
        n=1.337,
        mua=45.84957865,
        mus=35654.05549,
        pf=mc.mcpf.Hg(0.9)
    )

    # blood
    material_blood = mc.mcmaterial.Material(
        n=1.337,
        mua=23054.27,
        mus=9398.5,
        pf=mc.mcpf.Hg(0.9)
    )

    materials = mc.mcmaterial.Materials([
        material_surrounding,
        material_epidermis,
        material_dermis,
        material_blood
    ])

    # DEFINE PENCIL BEAM SOURCE
    source = mc.mcsource.Line()

    # DEFINE FLUENCE OBJECT
    fluence = mc.mcfluence.Fluence(
        xaxis=xaxis, 
        yaxis=yaxis, 
        zaxis=zaxis,
        mode='deposition'
    )

    # DEFINE MC OBJECT FOR MONTE CARLO SIMULATIONS AND ASSIGN MATERIALS
    mc_obj = mc.Mc(
        voxels=voxels,
        materials=materials,
        fluence=fluence,
        source=source,
        cl_devices=cl_device,
        options=mc_options,
        cl_build_options=cl_build_options
    )
    mc_obj.rmax = 0.025

    # get mesh
    z, y, x = voxels.meshgrid()

    # compute logical indices for material assignment
    r_vessel = 0.0002*0.5
    epidermis_mask = z <= 0.0001
    dermis_mask = z > 0.0001

    #vessel_depth = np.round(np.linspace(200e-6, 800e-6, 25), 6)
    vessel_depth = 500e-6
    vessel_mask = (x - 0.0)**2 + (z - vessel_depth)**2 <= r_vessel**2

    # assign the materials to voxels
    mc_obj.voxels.material.fill(0)
    mc_obj.voxels.material[epidermis_mask] = 1
    mc_obj.voxels.material[dermis_mask] = 2
    mc_obj.voxels.material[vessel_mask] = 3

    mc_obj.run(1) # a dummy run to build the kernel

    print('|' + '='*59)
    print('| Selected OpenCL device    :', cl_device.name)
    print('| Selected OpenCL platform  :', cl_device.platform.name)
    print('| MC kerenel stepping method:', kernel_method_name)
    print('| Survival lottery threshold:', lottery_threshold)
    print('| Materials placed in       : "{}" memory'.format(material_mem_name))
    print('|' + '-'*59)

    full_filename = output
    if not full_filename:
        full_filename = '2-layer-skin-{:.0f}um-vessel-{:.0f}um-depth-deposition'.format(
            r_vessel*1e6*2.0, vessel_depth*1e6).replace('.', '_') + '.npz'
    else:
        full_filename = os.path.splitext(filename)[0] + '.npz'

    progress_monitor = McProgressMonitor(mc_obj, num_packets)

    # RUN THE MONTE CARLO SIMULATIONS
    t_start = time.perf_counter()
    _, fluence_res, _ = mc_obj.run(num_packets, verbose=True)
    t_duration = time.perf_counter() - t_start
    print('|' + '-'*59)
    print('| Processed                 : {:,d} packets'.format(num_packets))
    print('| Duration                  : {:.3f} s'.format(t_duration))
    print('| Absorbed energy (%)       :', 
        100.0*fluence_res.data.sum()*voxel_size**3)

    if output:
        data = {
            'num_packets': num_packets,
            'batch_packets': num_packets,
            'voxels': voxels.todict(),
            'voxel_data': voxels.material,
            'materials': materials.todict(),
            'source': source.todict(),
            'detectors': None,
            'fluence': fluence.todict(),
            'fluence_data': fluence_res.data,
            'reflectance': None,
            'transmittance': None
        }

        print('| Writing results to:', full_filename)
        np.savez_compressed(full_filename, **data)
    else:
        print('| Results not saved. Specify a file with the "-o" option.')
    print('|' + '='*59)

    progress_monitor.terminate()
