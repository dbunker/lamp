#constant(obj, o2).
#constant(obj, o4).
#constant(obj, o6).
#constant(obj, o9).

#modeh(d1(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d1(o2)
}, {
  d1(o4), d1(o6), d1(o9)
}, {
  d3(o9).
  d4(o2).
  d5(o2).
  d7(o6).
  d7(o9).
  d8(o4).
}).