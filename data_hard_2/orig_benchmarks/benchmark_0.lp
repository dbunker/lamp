d11(o3).
d13(o3).
d10(o5).
d10(o2).
d12(o11).
d6(o0).
d4(o10).
d10(o11).
d3(o4).
d12(o5).

d7(X) :- d2(X), not d5(X).
d0(X) :- d4(X), not d10(X).
d7(X) :- d4(X), not d12(X).
d4(X) :- not d8(X), d10(X).
d1(X) :- d13(X), not d5(X).
d8(X) :- not d9(X), d5(X).
d4(X) :- not d10(X), d12(X).
d2(X) :- d0(X), not d3(X).
d11(X) :- not d0(X), d4(X).
d12(X) :- not d13(X), d0(X).
d10(X) :- not d0(X), d1(X).
d13(X) :- d8(X), not d2(X).
d11(X) :- d7(X), not d6(X).
d10(X) :- not d13(X), d9(X).
d3(X) :- not d8(X), d10(X).
d10(X) :- not d1(X), d13(X).
d0(X) :- d8(X), not d3(X).
d9(X) :- not d1(X), d11(X).
d12(X) :- not d10(X), d6(X).
d5(X) :- not d4(X), d0(X).