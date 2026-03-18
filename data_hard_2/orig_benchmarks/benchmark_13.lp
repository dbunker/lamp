d14(o9).
d12(o9).
d5(o9).
d3(o5).
d0(o13).
d17(o11).
d7(o9).
d10(o3).
d12(o10).
d8(o6).

d8(X) :- d5(X), d0(X), not d2(X).
d11(X) :- d7(X), d6(X), not d10(X).
d2(X) :- d0(X), d17(X), not d5(X).
d9(X) :- d8(X), not d2(X), d7(X).
d17(X) :- d14(X), d6(X), not d0(X).
d10(X) :- d3(X), not d13(X), d12(X).
d1(X) :- d0(X), not d4(X), d15(X).
d4(X) :- d14(X), d7(X), not d0(X).
d10(X) :- d14(X), d16(X), not d17(X).
d7(X) :- d6(X), not d4(X), d13(X).
d12(X) :- d4(X), not d11(X), d15(X).
d16(X) :- d14(X), d4(X), not d8(X).
d15(X) :- d14(X), d5(X), not d1(X).
d6(X) :- d5(X), not d9(X), d15(X).
d13(X) :- not d0(X), d3(X), d10(X).