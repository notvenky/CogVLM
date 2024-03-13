import mvp

model = mvp.load("vitb-mae-egosoup")
model.freeze()

import ipdb; ipdb.set_trace()