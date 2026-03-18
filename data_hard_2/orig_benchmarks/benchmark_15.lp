d6(o4).
d1(o4).
d15(o1).
d12(o6).
d17(o12).
d14(o1).
d16(o2).
d12(o3).
d12(o11).
d16(o3).
d9(o6).
d7(o4).
d10(o11).
d10(o0).
d4(o12).

d12(X) :- not d16(X), d6(X).
d14(X) :- not d0(X), d4(X).
d13(X) :- not d8(X), d7(X).
d7(X) :- d14(X), not d15(X).
d3(X) :- not d13(X), d0(X).
d3(X) :- d7(X), not d15(X).
d8(X) :- not d7(X), d13(X).
d7(X) :- not d9(X), d6(X).
d16(X) :- not d10(X), d13(X).
d3(X) :- not d12(X), d14(X).
d11(X) :- d7(X), not d9(X).
d0(X) :- d10(X), not d11(X).
d15(X) :- d17(X), not d6(X).
d15(X) :- d8(X), not d6(X).
d8(X) :- not d17(X), d4(X).
d1(X) :- not d6(X), d3(X).
d3(X) :- d0(X), not d11(X).
d17(X) :- not d3(X), d13(X).
d0(X) :- d4(X), not d6(X).
d15(X) :- d1(X), not d16(X).