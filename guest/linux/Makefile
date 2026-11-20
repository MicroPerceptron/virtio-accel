obj-m += virtio_accel.o
ccflags-y += -I$(src)/include/uapi
KDIR ?= /lib/modules/$(shell uname -r)/build
DESTDIR ?=
KVER = $(shell $(MAKE) -s -C $(KDIR) M=$(CURDIR) kernelrelease)

.PHONY: all clean install uninstall install-udev uninstall-udev

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

# Build/install for KDIR's kernel, which may differ from the build host's kernel.
install: all
	$(MAKE) -C $(KDIR) M=$(CURDIR) INSTALL_MOD_PATH="$(DESTDIR)" INSTALL_MOD_DIR=extra/virtio-accel modules_install
	@if [ -z "$(DESTDIR)" ]; then depmod -a "$(KVER)"; fi

uninstall:
	rm -f "$(DESTDIR)/lib/modules/$(KVER)/extra/virtio-accel/virtio_accel.ko" \
	    "$(DESTDIR)/lib/modules/$(KVER)/extra/virtio-accel/virtio_accel.ko.xz" \
	    "$(DESTDIR)/lib/modules/$(KVER)/extra/virtio-accel/virtio_accel.ko.gz" \
	    "$(DESTDIR)/lib/modules/$(KVER)/extra/virtio-accel/virtio_accel.ko.zst"
	@if [ -z "$(DESTDIR)" ]; then depmod -a "$(KVER)"; fi

install-udev:
	install -D -m 0644 70-virtio-accel.rules "$(DESTDIR)/etc/udev/rules.d/70-virtio-accel.rules"

uninstall-udev:
	rm -f "$(DESTDIR)/etc/udev/rules.d/70-virtio-accel.rules"
