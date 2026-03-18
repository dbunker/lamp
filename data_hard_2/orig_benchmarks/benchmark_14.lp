d9(o0).
d4(o7).
d1(o2).
d6(o3).
d7(o9).
d17(o12).
d5(o5).
d13(o11).
d5(o3).
d3(o10).

d12(X) :- not d2(X), d17(X), not d6(X).
d9(X) :- d14(X), not d6(X), not d12(X).
d0(X) :- d7(X), not d3(X), not d14(X).
d16(X) :- d5(X), not d2(X), not d10(X).
d2(X) :- d9(X), not d10(X), not d4(X).
d5(X) :- not d1(X), d10(X), not d9(X).
d10(X) :- not d0(X), not d11(X), d13(X).
d7(X) :- not d6(X), d2(X), not d5(X).
d14(X) :- not d0(X), not d9(X), d15(X).
d3(X) :- d7(X), not d16(X), not d17(X).
d5(X) :- not d3(X), not d2(X), d1(X).
d8(X) :- not d2(X), not d5(X), d11(X).
d11(X) :- d1(X), not d10(X), not d7(X).
d10(X) :- not d17(X), not d2(X), d3(X).
d15(X) :- d7(X), not d10(X), not d8(X).