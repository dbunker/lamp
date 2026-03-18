d16(o7).
d2(o4).
d16(o4).
d11(o1).
d10(o1).
d7(o7).
d15(o5).
d8(o13).
d6(o12).
d13(o2).
d11(o2).
d5(o0).
d11(o13).
d2(o3).
d10(o4).

d4(X) :- d9(X), not d16(X), d15(X).
d2(X) :- d7(X), not d12(X), d15(X).
d13(X) :- d5(X), d2(X), not d7(X).
d17(X) :- not d3(X), d10(X), d15(X).
d0(X) :- d7(X), not d4(X), d14(X).
d9(X) :- not d16(X), d12(X), d11(X).
d12(X) :- d14(X), not d1(X), d13(X).
d13(X) :- not d2(X), d17(X), d4(X).
d3(X) :- d9(X), d2(X), not d5(X).
d7(X) :- d13(X), not d17(X), d11(X).
d16(X) :- not d3(X), d6(X), d1(X).
d1(X) :- d16(X), d4(X), not d14(X).
d7(X) :- d9(X), d3(X), not d13(X).
d11(X) :- d14(X), d2(X), not d5(X).
d14(X) :- d7(X), not d8(X), d13(X).