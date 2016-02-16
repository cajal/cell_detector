FROM datajoint/datajoint-dev

MAINTAINER Fabian Sinz <sinz@bcm.edu>

WORKDIR /data


# install tools to compile
RUN \
  apt-get update && \
  apt-get install -y -q \
    build-essential && \
  apt-get update && \
  apt-get install -y -q \
    autoconf \
    automake \
    libtool \
    zlib1g-dev \
    wget

# Build HDF5
RUN cd ; wget https://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar.gz
RUN cd ; tar zxf hdf5-1.8.16.tar.gz
RUN cd ; mv hdf5-1.8.16 hdf5-setup
RUN cd ; cd hdf5-setup ; ./configure --prefix=/usr/local/
RUN cd ; cd hdf5-setup ; make -j 12 && make install

# cleanup
RUN cd ; rm -rf hdf5-setup
RUN apt-get -yq autoremove
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


COPY . /data/aod_cell_detection
RUN \
  pip install -e aod_cell_detection

#ENTRYPOINT ["worker"]
  
  
