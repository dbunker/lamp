d0(o7).
d1(o4).
d8(o6).
d5(o5).
d2(o0).
d9(o2).
d7(o6).
d3(o8).
d1(o3).

d6(X) :- not d5(X), d8(X).
d4(X) :- d7(X), not d1(X).
d3(X) :- d8(X), not d4(X).
d7(X) :- not d2(X), d6(X).
d8(X) :- d7(X), not d1(X).
d5(X) :- not d2(X), d1(X).
d3(X) :- not d1(X), d8(X).
d7(X) :- not d2(X), d6(X).
d1(X) :- not d8(X), d7(X).
d6(X) :- d1(X), not d0(X).
d2(X) :- d3(X), not d1(X).