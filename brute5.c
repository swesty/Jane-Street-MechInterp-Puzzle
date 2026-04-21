#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// Minimal MD5 implementation
static const uint32_t s[64] = {
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
    5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};
static const uint32_t K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,
    0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
    0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,
    0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,
    0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
    0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,
    0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,
    0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
    0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

#define LEFTROTATE(x,c) (((x)<<(c))|((x)>>(32-(c))))

void md5(const uint8_t *msg, size_t len, uint8_t digest[16]) {
    uint32_t h0=0x67452301, h1=0xefcdab89, h2=0x98badcfe, h3=0x10325476;
    size_t new_len = len + 1;
    while (new_len % 64 != 56) new_len++;
    uint8_t *buf = (uint8_t*)calloc(new_len + 8, 1);
    memcpy(buf, msg, len);
    buf[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    memcpy(buf + new_len, &bits, 8);
    for (size_t offset = 0; offset < new_len + 8; offset += 64) {
        uint32_t *M = (uint32_t*)(buf + offset);
        uint32_t A=h0, B=h1, C=h2, D=h3;
        for (int i = 0; i < 64; i++) {
            uint32_t F, g;
            if (i < 16)      { F = (B&C)|((~B)&D); g = i; }
            else if (i < 32) { F = (D&B)|((~D)&C); g = (5*i+1)%16; }
            else if (i < 48) { F = B^C^D;           g = (3*i+5)%16; }
            else              { F = C^(B|(~D));      g = (7*i)%16; }
            F = F + A + K[i] + M[g];
            A = D; D = C; C = B; B = B + LEFTROTATE(F, s[i]);
        }
        h0 += A; h1 += B; h2 += C; h3 += D;
    }
    free(buf);
    memcpy(digest,    &h0, 4);
    memcpy(digest+4,  &h1, 4);
    memcpy(digest+8,  &h2, 4);
    memcpy(digest+12, &h3, 4);
}

int main() {
    // target: c7ef65233c40aa32c2b9ace37595fa7c
    uint8_t target[16] = {0xc7,0xef,0x65,0x23,0x3c,0x40,0xaa,0x32,
                          0xc2,0xb9,0xac,0xe3,0x75,0x95,0xfa,0x7c};
    uint8_t digest[16];
    uint8_t buf[6];
    long long count = 0;

    // brute force 5-char printable ASCII (32..126)
    for (int a = 32; a < 127; a++)
    for (int b = 32; b < 127; b++)
    for (int c = 32; c < 127; c++)
    for (int d = 32; d < 127; d++)
    for (int e = 32; e < 127; e++) {
        buf[0]=a; buf[1]=b; buf[2]=c; buf[3]=d; buf[4]=e;
        md5(buf, 5, digest);
        if (digest[0]==target[0] && digest[1]==target[1] &&
            digest[2]==target[2] && digest[3]==target[3] &&
            memcmp(digest, target, 16)==0) {
            buf[5]=0;
            printf("FOUND: '%s'\n", buf);
            fflush(stdout);
            return 0;
        }
        count++;
        if (count % 500000000LL == 0) {
            fprintf(stderr, "checked %lld...\n", count);
        }
    }
    printf("not found in 5-char\n");

    // 6-char
    for (int a = 32; a < 127; a++)
    for (int b = 32; b < 127; b++) {
        for (int c = 32; c < 127; c++)
        for (int d = 32; d < 127; d++)
        for (int e = 32; e < 127; e++)
        for (int f = 32; f < 127; f++) {
            buf[0]=a; buf[1]=b; buf[2]=c; buf[3]=d; buf[4]=e; buf[5]=f;
            md5(buf, 6, digest);
            if (digest[0]==target[0] && digest[1]==target[1] &&
                digest[2]==target[2] && digest[3]==target[3] &&
                memcmp(digest, target, 16)==0) {
                char out[7]; memcpy(out,buf,6); out[6]=0;
                printf("FOUND: '%s'\n", out);
                fflush(stdout);
                return 0;
            }
        }
        count += 95*95*95*95;
        fprintf(stderr, "6-char: checked prefix '%c%c' (%lld total)\n", a, b, count);
    }
    printf("not found in 6-char\n");
    return 0;
}
