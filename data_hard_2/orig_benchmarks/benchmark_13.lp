d12(o10).
d4(o0).
d11(o4).
d7(o13).
d6(o12).
d12(o11).
d5(o10).
d0(o3).
d17(o10).
d0(o8).

d3(X) :- d2(X), not d0(X), d9(X).
d15(X) :- d4(X), not d6(X), d0(X).
d13(X) :- d17(X), not d3(X), d5(X).
d5(X) :- d4(X), d11(X), not d15(X).
d15(X) :- not d7(X), d17(X), d13(X).
d6(X) :- d1(X), not d5(X), d15(X).
d10(X) :- d12(X), not d1(X), d7(X).
d8(X) :- d1(X), d14(X), not d16(X).
d9(X) :- not d4(X), d16(X), d3(X).
d11(X) :- not d1(X), d4(X), d7(X).
d16(X) :- d4(X), not d14(X), d0(X).
d17(X) :- d2(X), d0(X), not d6(X).
d6(X) :- not d10(X), d12(X), d5(X).
d4(X) :- not d7(X), d14(X), d6(X).
d11(X) :- d8(X), d7(X), not d2(X).
d14(X) :- not d17(X), d15(X), d5(X).
d1(X) :- not d8(X), d12(X), d5(X).
d2(X) :- not d4(X), d12(X), d15(X).
d16(X) :- d2(X), not d14(X), d11(X).
d0(X) :- not d5(X), d14(X), d11(X).