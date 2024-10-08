// https://en.wikipedia.org/wiki/Crystal_radio#/media/File:Inductively_coupled_crystal_radio_circuit.svg

//https://notisrac.github.io/FileToCArray/
// array size is 1102
static const uint8_t crystal[]  = {
  0x42, 0x4d, 0x4e, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x28, 0x00, 
  0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x10, 0x04, 0x00, 0x00, 0x25, 0x16, 0x00, 0x00, 0x25, 0x16, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf8, 0x03, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf8, 0x03, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xe3, 0xff, 0xff, 0xff, 0xfe, 0x3f, 0xff, 0xff, 0xff, 0xc0, 0x00, 
  0x7f, 0xff, 0xff, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 0xff, 0xc0, 0x00, 
  0x7f, 0xff, 0xff, 0xdf, 0xff, 0xe3, 0xff, 0xff, 0xff, 0xfe, 0x3f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xff, 0x1f, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xfe, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xfd, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0x80, 
  0x00, 0x00, 0x7d, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xbe, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xbe, 0x1f, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xff, 
  0xfd, 0xff, 0xbe, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xff, 
  0xf8, 0xff, 0x3d, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x1f, 0xff, 0xff, 0xff, 
  0xf8, 0x00, 0x7d, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfe, 0x67, 0xff, 0xff, 0xff, 
  0xf8, 0xff, 0xbc, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfd, 0xf3, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xbe, 0x1f, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfd, 0xfb, 0xff, 0xff, 0xfb, 
  0xff, 0xff, 0xbe, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfb, 0xf9, 0xff, 0xff, 0xf1, 
  0xff, 0x1f, 0x3d, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xf9, 0xf8, 0x3f, 0xff, 0x80, 
  0xfe, 0x00, 0x7d, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfd, 0xfb, 0xdf, 0xff, 0xb1, 
  0xff, 0x1f, 0xbd, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfc, 0xf7, 0xef, 0xff, 0xb9, 
  0xff, 0xff, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x0f, 0xff, 0xff, 0x7f, 0xfe, 0x0f, 0xe7, 0xff, 0xbe, 
  0xff, 0xff, 0xbd, 0xff, 0xff, 0xf7, 0xff, 0xef, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xf7, 0xff, 0xbf, 
  0x6f, 0x1f, 0xbd, 0xff, 0xff, 0xf7, 0xff, 0xef, 0xff, 0xe0, 0x03, 0xff, 0xff, 0xf7, 0xff, 0xbf, 
  0xae, 0x00, 0x7d, 0xff, 0xf9, 0xf7, 0xff, 0xef, 0xff, 0xe0, 0x03, 0xff, 0xff, 0xf7, 0xff, 0xbf, 
  0xcf, 0x1f, 0x3e, 0xff, 0xfc, 0xf7, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf7, 0xcf, 0xbf, 
  0x0f, 0xbf, 0xbe, 0x1f, 0xfe, 0x77, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf7, 0xe7, 0xbf, 
  0xff, 0xff, 0xbc, 0xff, 0xff, 0x37, 0xff, 0xef, 0xff, 0xe0, 0x03, 0xff, 0xff, 0xf7, 0xf3, 0xbf, 
  0xf8, 0xff, 0xbd, 0xff, 0xfe, 0x00, 0x3f, 0xef, 0xff, 0xe0, 0x03, 0xff, 0x0f, 0xf7, 0xf9, 0xbf, 
  0xf8, 0x00, 0x7d, 0xff, 0xfe, 0x00, 0x3f, 0xef, 0xff, 0xff, 0x7f, 0xfe, 0xf7, 0xef, 0xf0, 0x01, 
  0xf8, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xef, 0xff, 0xff, 0x7f, 0xfd, 0xfb, 0xcf, 0xf0, 0x01, 
  0xfd, 0xff, 0xfe, 0x1f, 0xff, 0xff, 0xff, 0xef, 0xff, 0xff, 0x7f, 0xf9, 0xfb, 0x9f, 0xff, 0xff, 
  0xff, 0xff, 0xfe, 0xff, 0xfe, 0x00, 0x3f, 0xef, 0xff, 0xff, 0x7f, 0xfb, 0xf8, 0x3f, 0xff, 0xff, 
  0xff, 0xff, 0xfd, 0xff, 0xfe, 0x00, 0x3f, 0xef, 0xff, 0xff, 0x7f, 0xfd, 0xf9, 0xff, 0xf0, 0x01, 
  0xff, 0xff, 0xfd, 0xff, 0xff, 0xf4, 0xff, 0xef, 0xff, 0xff, 0x7f, 0xfd, 0xfb, 0xff, 0xf0, 0x01, 
  0xff, 0xff, 0xfe, 0xff, 0xff, 0xf6, 0x5f, 0xef, 0xff, 0xff, 0x7f, 0xfe, 0xf7, 0xff, 0xff, 0xa7, 
  0xff, 0xff, 0xff, 0x1f, 0xff, 0xf7, 0x1f, 0xef, 0xff, 0xff, 0x7f, 0xff, 0x0f, 0xff, 0xff, 0xb2, 
  0xff, 0xff, 0xff, 0xdf, 0xff, 0xf6, 0x1f, 0xef, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xb8, 
  0xff, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0x8f, 0xef, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xb0, 
  0xff, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0xef, 0xef, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbc, 
  0x7f, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0xff, 0xef, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0x7f, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0xff, 0xef, 0xcf, 0x3f, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0xff, 0xef, 0xc3, 0x3f, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0xff, 0xef, 0xc1, 0x3f, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xff, 0xdf, 0xff, 0xf7, 0xff, 0xef, 0xc0, 0x3e, 0x3f, 0xff, 0xbf, 0xff, 0xff, 0xbf, 
  0xff, 0xff, 0xff, 0xc0, 0x00, 0x07, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x3f, 0xff, 0xff, 0x1f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc0, 0x3e, 0x3f, 0xff, 0xff, 0xff, 0xfe, 0x0f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc1, 0x3f, 0x7f, 0xff, 0xff, 0xff, 0xfe, 0xaf, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc3, 0x3f, 0xff, 0xff, 0xff, 0xff, 0xfd, 0xb7, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xcf, 0x3f, 0xff, 0xff, 0xff, 0xff, 0xfd, 0xb7, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfb, 0xbb, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf3, 0xb9, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf7, 0xbd, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe7, 0xbc, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
/*
static const uint8_t crystal[]  = {
  0x42, 0x4d, 0x3e, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x28, 0x00, 
  0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x13, 0x0b, 0x00, 0x00, 0x13, 0x0b, 0x00, 0x00, 0x02, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xe1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xe3, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xc7, 0xff, 0xff, 0xff, 0xfe, 0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 0xff, 0xff, 0xff, 
  0xff, 0x00, 0x7f, 0x97, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9f, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xcf, 0xff, 0xff, 0xff, 0xfe, 0x3f, 0xff, 0x9f, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xff, 0xff, 
  0xf8, 0x00, 0x0f, 0xbf, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xff, 0xff, 
  0xff, 0xe7, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xff, 0xff, 
  0xff, 0xf7, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xff, 0xff, 
  0xff, 0xf7, 0xfe, 0x3f, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xff, 0xff, 
  0xff, 0xf0, 0xfc, 0x3f, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0x8f, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfe, 0x67, 0xff, 0xff, 0xff, 
  0xf1, 0xfe, 0x79, 0xbf, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfc, 0xfb, 0xff, 0xff, 0xff, 
  0xe0, 0x00, 0xfc, 0x3f, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfd, 0xf9, 0xff, 0xff, 0xff, 
  0xf1, 0xfe, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xf9, 0xfd, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xf9, 0xfc, 0x3f, 0xff, 0xff, 
  0xff, 0xff, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfd, 0xf9, 0x9f, 0xff, 0xff, 
  0xfc, 0x7e, 0x79, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xfc, 0xfb, 0xef, 0xff, 0xff, 
  0xfc, 0x00, 0xfc, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0x7f, 0xfe, 0x07, 0xe7, 0xff, 0xf7, 
  0xfc, 0x3e, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xdf, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xf7, 0xff, 0xe3, 
  0xfe, 0x7f, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xdf, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xf3, 0xff, 0x03, 
  0xff, 0xff, 0x7b, 0xff, 0xff, 0xef, 0xff, 0xdf, 0xff, 0xe0, 0x03, 0xff, 0xff, 0xf3, 0xff, 0x61, 
  0x7c, 0x7f, 0x79, 0xff, 0xff, 0xef, 0xff, 0xdf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf3, 0xff, 0x7c, 
  0x7c, 0x00, 0xfc, 0x3f, 0xf9, 0xef, 0xff, 0xdf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf3, 0xff, 0x7e, 
  0x3c, 0x3e, 0xf9, 0xff, 0xfc, 0xef, 0xff, 0xdf, 0xff, 0xe0, 0x83, 0xff, 0xff, 0xf3, 0xff, 0x7c, 
  0x3e, 0x7f, 0x7b, 0xff, 0xfe, 0x6f, 0xff, 0xdf, 0xff, 0xe0, 0x03, 0xff, 0x0f, 0xf7, 0xff, 0x7f, 
  0x1f, 0xff, 0x7b, 0xff, 0xff, 0x2f, 0xff, 0xdf, 0xff, 0xff, 0x7f, 0xfe, 0x73, 0xe7, 0xff, 0x7f, 
  0xf3, 0xff, 0x7b, 0xff, 0xfc, 0x00, 0x7f, 0xdf, 0xff, 0xff, 0x7f, 0xfd, 0xfb, 0xef, 0xff, 0x7f, 
  0xe0, 0x00, 0xfc, 0x3f, 0xff, 0xcf, 0xff, 0xdf, 0xff, 0xff, 0x7f, 0xf9, 0xf9, 0x9f, 0xff, 0x7f, 
  0xe0, 0x03, 0xf9, 0xff, 0xff, 0xe7, 0xff, 0xdf, 0xff, 0xff, 0x7f, 0xf9, 0xfc, 0x3f, 0xff, 0x7f, 
  0xf3, 0xff, 0xfb, 0xff, 0xfc, 0x00, 0x7f, 0xdf, 0xff, 0xff, 0x7f, 0xf9, 0xfc, 0xff, 0xcf, 0x7f, 
  0xff, 0xff, 0xfb, 0xff, 0xff, 0xe9, 0xff, 0xdf, 0xff, 0xff, 0x7f, 0xfd, 0xfb, 0xff, 0xe7, 0x7f, 
  0xff, 0xff, 0xfb, 0xff, 0xff, 0xec, 0xbf, 0xdf, 0xff, 0xff, 0x7f, 0xfc, 0xf3, 0xff, 0xf3, 0x7f, 
  0xff, 0xff, 0xfc, 0x3f, 0xff, 0xee, 0x3f, 0xdf, 0xff, 0xff, 0x7f, 0xfe, 0x07, 0xff, 0xfb, 0x7f, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xec, 0x3f, 0xdf, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xe0, 0x03, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0x1f, 0xdf, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xfe, 0xff, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0xcf, 0xdf, 0xff, 0xff, 0x7f, 0xff, 0x9f, 0xff, 0xff, 0x7f, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xdf, 0xff, 0xff, 0x7f, 0xff, 0xbf, 0xff, 0xe0, 0x03, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xdf, 0xbf, 0x3f, 0x7f, 0xff, 0xbf, 0xff, 0xe0, 0x4b, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xdf, 0x8f, 0x3f, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0x6f, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xdf, 0x83, 0x3f, 0x7f, 0xff, 0xbf, 0xff, 0xff, 0x71, 
  0xff, 0xff, 0xff, 0xbf, 0xff, 0xef, 0xff, 0xdf, 0x81, 0x3e, 0x3f, 0xff, 0xbf, 0xff, 0xff, 0x71, 
  0xff, 0xff, 0xff, 0x80, 0x00, 0x0f, 0xff, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x3f, 0xff, 0xff, 0x60, 
  0xff, 0xff, 0xff, 0xc0, 0x00, 0x0f, 0xff, 0xe0, 0x00, 0x3c, 0x10, 0x00, 0x7f, 0xff, 0xff, 0x7c, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x81, 0xbe, 0x3f, 0xff, 0xff, 0xff, 0xff, 0x7f, 
  0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x87, 0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x9f, 0x3f, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x7f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x3f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd, 0x1f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf9, 0x5f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfb, 0x4f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf3, 0x6f, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf7, 0x77, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe7, 0x73, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0x7b, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xdf, 0x79, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x9f, 0x7d, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80, 0x01, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
*/