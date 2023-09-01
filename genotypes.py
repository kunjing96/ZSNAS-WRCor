MobileNetV2 = \
('SuperConvK3BNRELU(3,32,2,1)' + \
 'SuperResIDWE1K3(32,16,1,32,1)' + \
 'SuperResIDWE6K3(16,24,2,16,2)' + \
 'SuperResIDWE6K3(24,32,2,24,3)' + \
 'SuperResIDWE6K3(32,64,2,32,4)' + \
 'SuperResIDWE6K3(64,96,1,64,3)' + \
 'SuperResIDWE6K3(96,160,2,96,3)' + \
 'SuperResIDWE6K3(160,320,1,160,1)' + \
 'SuperConvK1BNRELU(320,1280,1,1)',
 224)

zennet_flops400M_res192 = \
('SuperConvK3BNRELU(3,32,2,1)' + \
 'SuperResIDWE1K7(32,48,2,40,1)' + \
 'SuperResIDWE1K7(48,48,1,40,1)' + \
 'SuperResIDWE1K7(48,88,2,96,1)' + \
 'SuperResIDWE1K7(88,88,1,96,1)' + \
 'SuperResIDWE2K7(88,144,2,176,1)' + \
 'SuperResIDWE2K7(144,144,1,176,5)' + \
 'SuperResIDWE4K7(144,248,2,244,10)' + \
 'SuperConvK1BNRELU(248,1024,1,1)',
 192)

zennet_flops600M_res256 = \
('SuperConvK3BNRELU(3,16,2,1)' + \
 'SuperResIDWE1K7(16,32,2,32,1)' + \
 'SuperResIDWE1K7(32,32,1,32,1)' + \
 'SuperResIDWE1K7(32,72,2,96,1)' + \
 'SuperResIDWE1K7(72,72,1,96,1)' + \
 'SuperResIDWE2K7(72,120,2,128,8)' + \
 'SuperResIDWE4K7(120,200,2,176,6)' + \
 'SuperResIDWE4K7(200,168,1,192,10)' + \
 'SuperConvK1BNRELU(168,1536,1,1)',
 256)

zennet_flops900M_res224 = \
('SuperConvK3BNRELU(3,16,2,1)' + \
 'SuperResIDWE1K7(16,48,2,72,1)' + \
 'SuperResIDWE1K7(48,48,1,72,1)' + \
 'SuperResIDWE2K7(48,72,2,64,1)' + \
 'SuperResIDWE2K7(72,72,1,64,5)' + \
 'SuperResIDWE2K7(72,152,2,144,6)' + \
 'SuperResIDWE2K7(152,360,2,352,8)' + \
 'SuperResIDWE4K7(360,288,1,264,6)' + \
 'SuperConvK1BNRELU(288,2048,1,1)',
 224)

R_our_vote = \
('SuperConvK3BNRELU(3,48,2,1)' + \
 'SuperResIDWE6K5(48,40,1,40,2)' + \
 'SuperResIDWE6K3(40,32,1,48,3)' + \
 'SuperResIDWE6K7(32,32,2,24,3)' + \
 'SuperResIDWE6K7(32,96,1,40,2)' + \
 'SuperResIDWE2K7(96,240,2,72,6)' + \
 'SuperResIDWE1K3(240,168,2,192,6)' + \
 'SuperResIDWE4K7(168,192,2,152,1)' + \
 'SuperConvK1BNRELU(192,1440,1,1)',
 320)

RL_our_vote = \
('SuperConvK3BNRELU(3,48,2,1)' + \
 'SuperResIDWE4K7(48,48,1,48,2)' + \
 'SuperResIDWE4K5(48,40,1,48,3)' + \
 'SuperResIDWE2K3(40,64,2,48,2)' + \
 'SuperResIDWE1K5(64,48,1,40,2)' + \
 'SuperResIDWE4K7(48,208,2,96,6)' + \
 'SuperResIDWE2K3(208,96,2,208,5)' + \
 'SuperResIDWE4K3(96,216,2,160,1)' + \
 'SuperConvK1BNRELU(216,928,1,1)',
 320)

EA_our_vote = \
('SuperConvK3BNRELU(3,48,2,1)' + \
 'SuperResIDWE6K7(48,32,2,32,2)' + \
 'SuperResIDWE6K5(32,40,2,48,3)' + \
 'SuperResIDWE6K7(40,40,1,32,3)' + \
 'SuperResIDWE6K7(40,72,2,64,4)' + \
 'SuperResIDWE6K7(72,176,1,56,6)' + \
 'SuperResIDWE4K5(176,240,2,232,6)' + \
 'SuperResIDWE2K7(240,136,1,176,2)' + \
 'SuperConvK1BNRELU(136,424,1,1)',
 320)

R_our_vote_600M = \
('SuperConvK3BNRELU(3,40,2,1)' + \
 'SuperResIDWE4K5(40,32,2,24,2)' + \
 'SuperResIDWE6K5(32,16,2,24,3)' + \
 'SuperResIDWE2K7(16,48,2,40,3)' + \
 'SuperResIDWE2K5(48,56,1,48,4)' + \
 'SuperResIDWE2K7(56,224,2,72,5)' + \
 'SuperResIDWE6K3(224,184,1,192,6)' + \
 'SuperResIDWE6K7(184,128,1,144,2)' + \
 'SuperConvK1BNRELU(128,1240,1,1)',
 256)

RL_our_vote_600M = \
('SuperConvK3BNRELU(3,32,2,1)' + \
 'SuperResIDWE6K3(32,32,1,16,1)' + \
 'SuperResIDWE1K5(32,24,2,40,2)' + \
 'SuperResIDWE1K7(24,48,2,48,3)' + \
 'SuperResIDWE2K3(48,88,2,64,4)' + \
 'SuperResIDWE6K5(88,152,2,48,6)' + \
 'SuperResIDWE6K7(152,152,1,144,6)' + \
 'SuperResIDWE4K5(152,240,1,152,2)' + \
 'SuperConvK1BNRELU(240,1184,1,1)',
 192)

EA_our_vote_600M = \
('SuperConvK3BNRELU(3,48,2,1)' + \
 'SuperResIDWE2K7(48,32,2,40,2)' + \
 'SuperResIDWE2K7(32,32,2,48,3)' + \
 'SuperResIDWE6K7(32,64,1,40,3)' + \
 'SuperResIDWE6K7(64,88,2,40,4)' + \
 'SuperResIDWE6K7(88,104,1,56,6)' + \
 'SuperResIDWE6K5(104,152,2,192,6)' + \
 'SuperResIDWE6K5(152,240,1,136,2)' + \
 'SuperConvK1BNRELU(240,1264,1,1)',
 224)
