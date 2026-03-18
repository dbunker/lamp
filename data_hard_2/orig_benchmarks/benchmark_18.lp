d5(o13).
d13(o16).
d14(o5).
d17(o7).
d0(o13).
d12(o16).
d6(o0).
d17(o3).
d3(o1).
d15(o13).

d10(X) :- not d15(X), d13(X).
d4(X) :- d14(X), not d2(X).
d9(X) :- d8(X), not d3(X).
d4(X) :- d8(X), not d5(X).
d16(X) :- not d3(X), d12(X).
d0(X) :- d8(X), not d5(X).
d17(X) :- not d0(X), d6(X).
d8(X) :- d12(X), not d5(X).
d2(X) :- not d0(X), d13(X).
d9(X) :- d0(X), not d4(X).
d12(X) :- d8(X), not d6(X).
d1(X) :- not d17(X), d13(X).
d10(X) :- d14(X), not d2(X).
d3(X) :- d8(X), not d5(X).
d17(X) :- not d1(X), d16(X).