d0(o5).
d4(o0).
d4(o4).
d0(o1).
d5(o2).
d4(o3).

d1(X) :- not d2(X), d0(X), not d3(X).
d5(X) :- not d2(X), d3(X), not d0(X).
d5(X) :- not d2(X), d3(X), not d1(X).
d3(X) :- not d2(X), d5(X), not d0(X).
d5(X) :- not d2(X), d0(X), not d1(X).
d1(X) :- not d2(X), not d0(X), d4(X).
d0(X) :- not d3(X), d2(X), not d1(X).
d1(X) :- not d5(X), not d0(X), d4(X).
d1(X) :- d2(X), not d5(X), not d0(X).
d2(X) :- not d3(X), d1(X), not d4(X).
d2(X) :- d0(X), not d1(X), not d4(X).