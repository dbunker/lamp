d2(o7).
d1(o0).
d1(o9).
d7(o2).
d4(o8).
d8(o5).
d3(o5).
d9(o3).
d6(o8).

d9(X) :- not d8(X), d0(X), d6(X).
d7(X) :- d0(X), d6(X), not d5(X).
d6(X) :- d9(X), not d7(X), d1(X).
d8(X) :- not d7(X), d6(X), d1(X).
d5(X) :- d9(X), not d3(X), d7(X).
d8(X) :- d0(X), not d7(X), d5(X).
d8(X) :- not d2(X), d6(X), d4(X).
d3(X) :- d2(X), not d7(X), d5(X).
d0(X) :- d2(X), not d6(X), d4(X).
d5(X) :- d9(X), d6(X), not d1(X).
d5(X) :- d9(X), not d8(X), d4(X).