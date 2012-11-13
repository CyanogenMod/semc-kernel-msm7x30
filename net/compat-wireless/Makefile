NOSTDINC_FLAGS := -I$(srctree)/net/compat-wireless/include/ \
	-include $(srctree)/net/compat-wireless/include/linux/compat-2.6.h \
	-DCOMPAT_STATIC
dummy=$(shell (cd $(KBUILD_SRC)/net/compat-wireless; ./scripts/check_config.sh))

obj-$(CONFIG_COMPAT_WIRELESS) += net/wireless/
obj-$(CONFIG_COMPAT_WIRELESS) += net/mac80211/
obj-$(CONFIG_COMPAT_WIRELESS) += compat/
obj-$(CONFIG_COMPAT_WIRELESS_MODULES) += drivers/
